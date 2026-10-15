"""Batched OpenVLA-OFT forward: the same computation as the reference, at B > 1.

The reference path is `get_vla_action` (openvla-oft/experiments/robot/openvla_utils.py
line 715) calling `OpenVLAForActionPrediction.predict_action`
(prismatic/extern/hf/modeling_prismatic.py line 944). Almost all of it is already
batch-shaped -- (B, seq_len, D) throughout, and `_process_proprio_features` even reshapes
proprio to (bsz, -1). Two places assume B == 1, both found by probing rather than by
reading hopefully (see bench/out/m3_probe_batching.json):

  1. `predict_action` appends the empty token id 29871 when the prompt does not already
     end with it, with a hard-coded shape (1, 1):
         torch.cat((input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), 0)), dim=1)
     The LIBERO prompt ends with 29901 (':'), so the branch always fires, and at B > 1 it
     raises "Sizes of tensors must match except in dimension 1".
     Handled by `append_empty_token`: we append it ourselves for the whole batch, which
     makes `torch.all(input_ids[:, -1] == 29871)` true so the branch is skipped. The
     sequence is byte-identical to what the single-request path builds.

  2. `_regression_or_discrete_prediction` ends with
         normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
     i.e. (8, 7), which at B > 1 raises "shape '[8, 7]' is invalid for input of size 112".
     Handled by `install_reshape_patch()`, which rebinds that one method on the model
     class with the same body and `reshape(-1, NUM_ACTIONS_CHUNK, ACTION_DIM)`. Nothing
     else in the function changes. The research tree is not modified.

At B == 1 the only difference from the reference is that actions come back as (1, 8, 7)
instead of (8, 7); the elements are the same ones in the same order, which is what the
parity check verifies.

Memory: the 7B bf16 weights occupy ~20 GB of the 24 GB card, so the usable batch is
small -- B = 16 OOMs. The server caps B accordingly.
"""

import numpy as np
import torch

EMPTY_TOKEN_ID = 29871
_PATCHED = {"done": False}
# M4 switches read by the patched _regression_or_discrete_prediction below.
SKIP_LM_HEAD = {"on": False}
CUDA_GRAPHS = {"on": False, "store": {}, "pool": None}


def assert_uniform_prompt_length(prepared):
    """Refuse a batch whose prompts differ in length, and say why.

    The prompt is "In: What action should the robot take to {instruction}?", so its token
    count varies with the task: the six LIBERO-Safety instructions tokenise to 24, 28, 29,
    30, 32 and 32 tokens. torch.cat along the batch dimension needs equal lengths, so a
    mixed batch raises "Sizes of tensors must match except in dimension 0".

    Left-padding to the longest prompt makes it RUN, and was tried and rejected: it is
    numerically wrong here. predict_action calls the language model with
    position_ids=None, so Llama derives positions as 0..L-1 for every row; left-padding a
    shorter prompt therefore shifts all of its real tokens' rotary positions. Measured
    cost of doing it anyway (bench/out/m3_mixed_batch.json): worst |batched - single|
    action difference 3.1e-02 at B=2 and 3.2e-01 at B=3 and B=6, against 4.8e-03 to
    6.6e-03 for a same-length batch. 0.32 on an action in roughly [-1, 1] is not rounding.

    Doing it properly would mean deriving position_ids from the attention mask and passing
    them through predict_action, which that function does not accept -- a deeper fork of
    the reference than M3 should carry for a mechanism worth 1.27x.

    So the server batches only requests that share a prompt length (it groups by it), and
    this assertion is the guarantee that it did. It cannot fire in production; if it does,
    it is loud instead of silently wrong.
    """
    lens = sorted({int(p["input_ids"].shape[1]) for p in prepared})
    if len(lens) > 1:
        raise ValueError(
            f"batch mixes prompt lengths {lens}; the server must group by prompt length "
            "(see assert_uniform_prompt_length.__doc__ for why padding is not the answer)"
        )
    return lens[0]


def append_empty_token(input_ids, attention_mask):
    """Append id 29871 to every row, so predict_action's batch-1 branch never fires."""
    b = input_ids.shape[0]
    tok = torch.full((b, 1), EMPTY_TOKEN_ID, dtype=input_ids.dtype, device=input_ids.device)
    one = torch.ones((b, 1), dtype=attention_mask.dtype, device=attention_mask.device)
    return torch.cat([input_ids, tok], dim=1), torch.cat([attention_mask, one], dim=1)


def _run_trunk_graphed(trunk, embeds, attn):
    """Replay a captured CUDA graph for this sequence length, capturing on first sight."""
    import torch

    key = (int(embeds.shape[0]), int(embeds.shape[1]))
    store = CUDA_GRAPHS["store"]
    entry = store.get(key)

    def _eager():
        return trunk(
            input_ids=None,
            attention_mask=attn,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state

    if entry is None:
        # one eager call to settle the allocator, then capture against static buffers
        out = _eager()
        se = embeds.clone()
        sa = attn.clone()
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        try:
            with torch.cuda.graph(g, pool=CUDA_GRAPHS["pool"]):
                so = trunk(
                    input_ids=None,
                    attention_mask=sa,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=se,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                ).last_hidden_state
            store[key] = dict(graph=g, embeds=se, attn=sa, out=so)
        except Exception as e:  # capture can fail; fall back and say so once
            store[key] = dict(graph=None, error=f"{type(e).__name__}: {e}")
            print(f"[GRAPH] capture failed for {key}: {e}; running eagerly", flush=True)
        return out

    if entry.get("graph") is None:
        return _eager()
    entry["embeds"].copy_(embeds)
    entry["attn"].copy_(attn)
    entry["graph"].replay()
    return entry["out"].clone()


def install_reshape_patch(model, force=False):
    """Rebind _regression_or_discrete_prediction with a batch-shaped final reshape.

    Copied from prismatic/extern/hf/modeling_prismatic.py lines 877-942 with exactly one
    change, marked below. Copied rather than imported because the batch-1 reshape is in
    the middle of the function and there is no hook around it; the research file itself
    must not be edited.
    """
    if _PATCHED["done"] and not force:
        return
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

    def _regression_or_discrete_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
    ):
        all_actions_mask = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )
        if SKIP_LM_HEAD["on"]:
            # The regression path uses only the final hidden state; LlamaForCausalLM
            # would also run lm_head over every position and we would discard it.
            trunk = self.language_model.model
            if CUDA_GRAPHS["on"]:
                last_hidden_states = _run_trunk_graphed(
                    trunk, multimodal_embeddings, multimodal_attention_mask
                )
            else:
                last_hidden_states = trunk(
                    input_ids=None,
                    attention_mask=multimodal_attention_mask,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=multimodal_embeddings,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                ).last_hidden_state
            language_model_output = None
        else:
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden_states = language_model_output.hidden_states[-1]
        actions_hidden_states = last_hidden_states[
            :,
            NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES
            + NUM_PROMPT_TOKENS
            + ACTION_DIM * NUM_ACTIONS_CHUNK,
            :,
        ]
        if action_head is not None:
            normalized_actions = action_head.predict_action(actions_hidden_states)
            # THE ONE CHANGE: reference does .reshape(NUM_ACTIONS_CHUNK, ACTION_DIM),
            # which is (8, 7) and therefore B == 1 only.
            normalized_actions = normalized_actions.reshape(-1, NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()
        else:
            predicted_action_token_ids = (
                language_model_output.logits[
                    :,
                    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES
                    + NUM_PROMPT_TOKENS
                    + ACTION_DIM * NUM_ACTIONS_CHUNK,
                ]
                .argmax(dim=2)
                .cpu()
                .numpy()
            )
            discretized_actions = self.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(
                discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
            )
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(-1, NUM_ACTIONS_CHUNK, ACTION_DIM)
        return normalized_actions, actions_hidden_states

    type(model)._regression_or_discrete_prediction = _regression_or_discrete_prediction
    _PATCHED["done"] = True


class BatchedPolicy:
    """Holds one copy of the policy and runs batched forwards.

    `prepare_request` does the CPU-side work (image crop/normalise, tokenise) and
    `forward` does one GPU forward for a list of prepared requests. They are separate so a
    server can decide where each runs; M3a does both on the server process, because the
    processor output for one request is 2 x 3 x 224 x 224 bf16 = 1.2 MB against 393 KB for
    the raw uint8 images, so moving it to the worker would increase bytes on the wire.
    """

    def __init__(self, cfg, model, processor, action_head, proprio_projector, device="cuda:0"):
        self.cfg = cfg
        self.model = model
        self.processor = processor
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.device = device
        install_reshape_patch(model)
        self._prompt_cache = {}

    def prompt_for(self, task_label):
        """The reference builds this per call; it depends only on the instruction."""
        p = self._prompt_cache.get(task_label)
        if p is None:
            p = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
            self._prompt_cache[task_label] = p
        return p

    def prepare_request(self, full_image, wrist_image, state, task_label):
        """CPU side: exactly what get_vla_action does before predict_action."""
        from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla

        all_images = prepare_images_for_vla([full_image, wrist_image], self.cfg)
        primary = all_images.pop(0)
        prompt = self.prompt_for(task_label)
        inputs = self.processor(prompt, primary)
        pix = [inputs["pixel_values"]]
        for w in all_images:
            pix.append(self.processor(prompt, w)["pixel_values"])
        proprio = state
        if self.cfg.use_proprio:
            stats = self.model.norm_stats[self.cfg.unnorm_key]["proprio"]
            proprio = normalize_proprio(state, stats)
        return dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=torch.cat(pix, dim=1),
            proprio=np.asarray(proprio),
        )

    @torch.inference_mode()
    def forward(self, prepared):
        """One GPU forward for a list of prepared requests -> list of (8, 7) float64."""
        assert_uniform_prompt_length(prepared)
        input_ids = torch.cat([p["input_ids"] for p in prepared], dim=0).to(self.device)
        attn = torch.cat([p["attention_mask"] for p in prepared], dim=0).to(self.device)
        pix = torch.cat([p["pixel_values"] for p in prepared], dim=0).to(
            self.device, dtype=torch.bfloat16
        )
        proprio = np.stack([np.asarray(p["proprio"]).reshape(-1) for p in prepared], axis=0)
        input_ids, attn = append_empty_token(input_ids, attn)
        actions, _hidden = self.model.predict_action(
            input_ids=input_ids,
            attention_mask=attn,
            pixel_values=pix,
            unnorm_key=self.cfg.unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=self.proprio_projector,
            action_head=self.action_head,
            use_film=False,
        )
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 2:  # B == 1 without the reshape patch
            actions = actions[None, ...]
        return [actions[i] for i in range(actions.shape[0])]
