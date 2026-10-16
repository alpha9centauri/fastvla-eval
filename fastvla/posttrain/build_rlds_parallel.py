# Copied from vla-safety-interp/src/build_rlds_parallel.py at commit e602fd1 (that tree is
# read-only for this project). Unmodified: it takes every path as an argument already.
#
#!/usr/bin/env python3
"""Parallel TFDS/RLDS build for LIBERO-Safety.

Each worker serializes a disjoint slice of episodes straight into its own
TFRecord shards using the builder's own feature spec, so no worker needs the
others. A single final pass records shard lengths and byte counts through TFDS's
own DatasetInfo writer, which is what makes the result a legitimate TFDS build
rather than a pile of TFRecords.

dataset_statistics is deliberately NOT written here: OpenVLA's loader computes it
over the whole dataset on first load and caches it under a hash of builder.info.
Precomputing it would risk disagreeing with what the loader would derive.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

NAME = "libero_safety"
VERSION = "1.0.0"


def episode_split(converter, max_episodes, stratified, val_split):
    """Reproduce build_rlds()'s split exactly so parallel == single-process."""
    indices = [ep["episode_index"] for ep in converter.episodes]
    if max_episodes and stratified:
        step = max(1, len(indices) // max_episodes)
        indices = indices[::step][:max_episodes]
    elif max_episodes:
        indices = indices[:max_episodes]
    val_count = int(len(indices) * val_split)
    return indices[: len(indices) - val_count], indices[len(indices) - val_count :]


def chunk(seq, n):
    """Contiguous chunks; preserves order so shard k holds a known episode range."""
    if not seq:
        return []
    size, rem = divmod(len(seq), n)
    out, i = [], 0
    for k in range(n):
        c = size + (1 if k < rem else 0)
        if c:
            out.append(seq[i : i + c])
        i += c
    return out


def run_worker(args_tuple):
    """Spawn one worker process; returns its shard manifest."""
    data_root, out_dir, split, shard_idx, num_shards, indices, meta_root = args_tuple
    spec = {
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "split": split,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "indices": indices,
        "meta_root": str(meta_root) if meta_root else None,
    }
    spec_path = Path(out_dir) / f"_spec_{split}_{shard_idx:05d}.json"
    spec_path.write_text(json.dumps(spec))

    # TFDS's Image feature encodes via a TF eager runner, which will claim a GPU
    # context if one is visible and can fail outright when another job holds the
    # card. Encoding is CPU work, so hide the GPUs from workers entirely.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3")
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", str(spec_path)],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker {split}/{shard_idx} failed rc={proc.returncode}\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    result = json.loads((Path(out_dir) / f"_done_{split}_{shard_idx:05d}.json").read_text())
    spec_path.unlink(missing_ok=True)
    return result


def worker_main(spec_path):
    """Serialize one shard. Runs in its own process with its own TF import."""
    spec = json.loads(Path(spec_path).read_text())
    sys.path.insert(0, str(Path(__file__).parent))

    import tensorflow as tf
    import libero_safety_converter_tfds as C

    conv = C.LiberoSafetyConverter(
        Path(spec["data_root"]), Path(spec["out_dir"]),
        Path(spec["meta_root"]) if spec["meta_root"] else None,
    )
    builder = C.LiberoSafety(data_dir=spec["out_dir"])
    features = builder.info.features

    split, idx, total = spec["split"], spec["shard_idx"], spec["num_shards"]
    shard = Path(spec["out_dir"]) / f"{NAME}-{split}.tfrecord-{idx:05d}-of-{total:05d}"

    n_written = 0
    skipped = []
    with tf.io.TFRecordWriter(str(shard)) as w:
        for episode_index in spec["indices"]:
            record = conv.process_episode(int(episode_index))
            if record is None:
                skipped.append(int(episode_index))
                continue
            instruction = record["episode_metadata"]["language_instruction"]
            suite = record["episode_metadata"]["suite"]
            n = len(record["steps"])
            steps = [
                {
                    "observation": {
                        "image": s["observation"]["image"],
                        "wrist_image": s["observation"]["wrist_image"],
                        "state": s["observation"]["state"].astype("float32"),
                    },
                    "action": s["action"].astype("float32"),
                    "language_instruction": instruction,
                    "is_first": t == 0,
                    "is_last": t == n - 1,
                    "is_terminal": t == n - 1,
                }
                for t, s in enumerate(record["steps"])
            ]
            w.write(features.serialize_example({
                "steps": steps,
                "episode_metadata": {
                    "episode_index": int(episode_index),
                    "suite": suite,
                    "language_instruction": instruction,
                },
            }))
            n_written += 1

    out = {
        "split": split, "shard_idx": idx, "num_examples": n_written,
        "num_bytes": shard.stat().st_size, "skipped": skipped,
        "path": str(shard),
    }
    (Path(spec["out_dir"]) / f"_done_{split}_{idx:05d}.json").write_text(json.dumps(out))
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--meta-root", type=Path, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--stratified", action="store_true")
    p.add_argument("--val-split", type=float, default=0.1)
    args = p.parse_args()

    if args.worker:
        return worker_main(args.worker)

    sys.path.insert(0, str(Path(__file__).parent))
    import tensorflow_datasets as tfds
    import libero_safety_converter_tfds as C

    out_dir = args.output_root / NAME / VERSION
    out_dir.mkdir(parents=True, exist_ok=True)

    conv = C.LiberoSafetyConverter(args.data_root, args.output_root, args.meta_root)
    train_idx, val_idx = episode_split(conv, args.max_episodes, args.stratified, args.val_split)
    print(f"episodes: {len(train_idx)} train / {len(val_idx)} validation", flush=True)

    jobs = []
    for split, idxs in (("train", train_idx), ("validation", val_idx)):
        parts = chunk(idxs, args.workers)
        for k, part in enumerate(parts):
            jobs.append((args.data_root, out_dir, split, k, len(parts), part, args.meta_root))
    print(f"shards: {sum(1 for j in jobs if j[2]=='train')} train, "
          f"{sum(1 for j in jobs if j[2]=='validation')} validation "
          f"({args.workers} concurrent workers)", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    results, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(run_worker, jobs):
            results.append(res)
            done += 1
            print(f"  [{done}/{len(jobs)}] {res['split']} shard {res['shard_idx']:05d}: "
                  f"{res['num_examples']} episodes, {res['num_bytes']/1e9:.2f} GB", flush=True)

    skipped = [e for r in results for e in r["skipped"]]
    if skipped:
        print(f"WARNING: {len(skipped)} episodes skipped: {skipped[:20]}", flush=True)

    # Shard order must match filename order for TFDS to index correctly.
    split_infos = []
    for split in ("train", "validation"):
        rs = sorted([r for r in results if r["split"] == split], key=lambda r: r["shard_idx"])
        if not rs:
            continue
        split_infos.append(tfds.core.SplitInfo(
            name=split,
            shard_lengths=[r["num_examples"] for r in rs],
            num_bytes=sum(r["num_bytes"] for r in rs),
        ))
        print(f"{split}: {sum(r['num_examples'] for r in rs)} examples "
              f"across {len(rs)} shards", flush=True)

    # Let TFDS write its own metadata so the result is a real build. The builder
    # is instantiated against a clean temp dir: pointing it at out_dir would make
    # it try to *read* a dataset_info.json that does not exist yet.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        builder = C.LiberoSafety(data_dir=tmp)
        builder.info.set_splits(tfds.core.SplitDict(split_infos))
        builder.info.write_to_directory(str(out_dir))

    for f in out_dir.glob("_done_*.json"):
        f.unlink()

    print(f"\nbuild complete in {(time.time()-t0)/60:.1f} min -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
