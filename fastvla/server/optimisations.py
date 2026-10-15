"""M4 step 2 changes to the forward, each switchable and measured on its own."""


def enable_prompt_cache(pol):
    """2a: tokenise each distinct instruction once, and skip the tokeniser per request.

    prepare_request calls processor(prompt, image) once per camera, and the combined
    processor tokenises the prompt on each call -- twice per request, for one of only five
    distinct instructions in the benchmark. This caches the token ids per instruction and
    routes the images through `processor.image_processor` alone, so the tokeniser does not
    run at all on the hot path.

    A first attempt cached the ids but still called the combined processor and overwrote
    the result afterwards, which saved nothing and measured as exactly nothing
    (prepare 11.33 ms against a baseline of 11.34 ms). This version actually removes the
    work; `verify_prompt_cache` checks the pixel values and ids are identical to the
    combined path before it is trusted.
    """
    import numpy as np
    from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
    from PIL import Image

    cache = {}

    def _tokens(task_label):
        if task_label not in cache:
            blank = Image.fromarray(np.zeros((224, 224, 3), np.uint8))
            out = pol.processor(pol.prompt_for(task_label), blank)
            cache[task_label] = (out["input_ids"], out["attention_mask"])
        return cache[task_label]

    def prepare_request(full_image, wrist_image, state, task_label):
        import torch

        all_images = prepare_images_for_vla([full_image, wrist_image], pol.cfg)
        pix = [
            pol.processor.image_processor(img, return_tensors="pt")["pixel_values"]
            for img in all_images
        ]
        ids, msk = _tokens(task_label)
        proprio = state
        if pol.cfg.use_proprio:
            stats = pol.model.norm_stats[pol.cfg.unnorm_key]["proprio"]
            proprio = normalize_proprio(state, stats)
        return dict(
            input_ids=ids,
            attention_mask=msk,
            pixel_values=torch.cat(pix, dim=1),
            proprio=np.asarray(proprio),
        )

    pol.prepare_request = prepare_request
    pol._prompt_token_cache = cache
    return "pretok"


def verify_prompt_cache(pol, full_image, wrist_image, state, task_label):
    """Prove the cached path produces byte-identical inputs to the combined processor."""
    import torch

    from fastvla.server.batched_forward import BatchedPolicy

    ref = BatchedPolicy.prepare_request(pol, full_image, wrist_image, state, task_label)
    got = pol.prepare_request(full_image, wrist_image, state, task_label)
    return dict(
        input_ids_equal=bool(torch.equal(ref["input_ids"], got["input_ids"])),
        attention_mask_equal=bool(torch.equal(ref["attention_mask"], got["attention_mask"])),
        pixel_values_equal=bool(torch.equal(ref["pixel_values"], got["pixel_values"])),
        pixel_max_abs_diff=float(
            (ref["pixel_values"].float() - got["pixel_values"].float()).abs().max()
        ),
        proprio_max_abs_diff=float(abs(ref["proprio"] - got["proprio"]).max()),
    )


def enable_compile(pol, mode="default"):
    """2d: torch.compile the LLM trunk."""
    import torch

    trunk = pol.model.language_model
    pol.model.language_model = torch.compile(trunk, mode=mode, dynamic=True)
    return f"compile:{mode}"


def enable_skip_lm_head(pol):
    """Found in the step-1 profile: the LM head is computed and thrown away.

    The L1-regression path reads only `language_model_output.hidden_states[-1]`; it never
    touches `.logits`. But `self.language_model` is a LlamaForCausalLM, so it runs
    lm_head over every position: about 600 tokens x 4096 hidden x 32k vocab x 2 flops
    = 158 GFLOP per forward, discarded. Calling the inner LlamaModel instead returns the
    same final hidden state without the head.

    Bit-exact by construction: this removes a computation whose result was dropped. The
    discrete-token branch does need logits, so it keeps the full causal LM.
    """
    from fastvla.server import batched_forward as BF

    BF.SKIP_LM_HEAD["on"] = True
    BF.install_reshape_patch(pol.model, force=True)
    return "skip_lm_head"


def enable_cuda_graphs(pol, lengths=(24, 28, 29, 30, 32)):
    """2e: capture one CUDA graph per prompt length for the LLM trunk.

    The trunk's input shapes are fixed once the prompt length is: the sequence is
    [BOS, 512 vision patches, 1 proprio, prompt-1, 57 action tokens], so only the prompt
    contributes variation, and the benchmark has five distinct instructions. One graph per
    length therefore covers every request with no padding -- which matters because M3
    measured that padding this model is numerically wrong (RoPE positions shift).

    Capture is lazy: the first request at a given length runs eagerly, is replayed once to
    warm the allocator, then captured. A length never seen is simply run eagerly.
    """
    import torch

    from fastvla.server import batched_forward as BF

    BF.CUDA_GRAPHS["on"] = True
    BF.CUDA_GRAPHS["store"] = {}
    BF.CUDA_GRAPHS["pool"] = torch.cuda.graph_pool_handle()
    BF.install_reshape_patch(pol.model, force=True)
    return f"cudagraph:{','.join(str(x) for x in lengths)}"


def enable_gpu_preprocessing(pol):
    """2b: normalise the camera images on the GPU instead of on the server CPU.

    The vision backbone is fused (SigLIP + DINOv2), so one image becomes SIX channels: the
    processor normalises it twice with different mean/std and concatenates. The constants
    are read from the processor's own `tvf_normalize_params` rather than retyped, so there
    is one source of truth, and `verify_gpu_preprocessing` checks the result against the
    CPU path before it is trusted.

    Implemented and measured rather than assumed, but the prediction from M3 is that it
    loses: the GPU is the bottleneck at 85% utilisation while 64 CPU cores sit in surplus,
    and the server already overlaps CPU preprocessing with the GPU forward in its receiver
    thread, so the CPU cost is hidden rather than additive.
    """
    import numpy as np
    import torch
    from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
    from PIL import Image

    ip = pol.processor.image_processor
    params = list(ip.tvf_normalize_params)
    norms = [
        (
            torch.tensor(prm["mean"], dtype=torch.float32, device=pol.device).view(1, 3, 1, 1),
            torch.tensor(prm["std"], dtype=torch.float32, device=pol.device).view(1, 3, 1, 1),
        )
        for prm in params
    ]
    cache = {}

    def _tokens(task_label):
        if task_label not in cache:
            blank = Image.fromarray(np.zeros((224, 224, 3), np.uint8))
            out = pol.processor(pol.prompt_for(task_label), blank)
            cache[task_label] = (out["input_ids"], out["attention_mask"])
        return cache[task_label]

    def _pixels(img):
        a = np.asarray(img, dtype=np.uint8)
        t = torch.from_numpy(a).to(pol.device, non_blocking=True)
        t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        chans = [((t - m) / sd) for m, sd in norms]
        return torch.cat(chans, dim=1).to(torch.bfloat16)

    def prepare_request(full_image, wrist_image, state, task_label):
        imgs = prepare_images_for_vla([full_image, wrist_image], pol.cfg)
        pix = torch.cat([_pixels(im) for im in imgs], dim=1)
        ids, msk = _tokens(task_label)
        proprio = state
        if pol.cfg.use_proprio:
            stats = pol.model.norm_stats[pol.cfg.unnorm_key]["proprio"]
            proprio = normalize_proprio(state, stats)
        return dict(
            input_ids=ids, attention_mask=msk, pixel_values=pix, proprio=np.asarray(proprio)
        )

    pol.prepare_request = prepare_request
    return "gpu_preproc"


def verify_gpu_preprocessing(pol, full_image, wrist_image, state, task_label):
    """Compare the GPU-normalised pixels against the processor's CPU result."""
    import torch

    from fastvla.server.batched_forward import BatchedPolicy

    ref = BatchedPolicy.prepare_request(pol, full_image, wrist_image, state, task_label)
    got = pol.prepare_request(full_image, wrist_image, state, task_label)
    a = ref["pixel_values"].float().cpu()
    b = got["pixel_values"].float().cpu()
    return dict(
        shape_ref=list(a.shape),
        shape_got=list(b.shape),
        shapes_equal=bool(a.shape == b.shape),
        max_abs_diff=(float((a - b).abs().max()) if a.shape == b.shape else None),
        exact=(bool(torch.equal(a, b)) if a.shape == b.shape else False),
    )
