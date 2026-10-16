# Copied from /data_new/vchandr2/libero_safety_converter_tfds.py (a loose file on the box,
# not under version control anywhere -- no commit to cite; md5 recorded in
# docs/POSTTRAIN.md). Unmodified except: the three hardcoded /data_new paths now default
# from fastvla/posttrain/config.env, so this runs somewhere else without editing the file.
#
#!/usr/bin/env python3
"""
LIBERO-Safety → TFDS/RLDS Converter (Production Format)

Converts LeRobot parquet + MP4 dataset to TFDS format for finetune.py.

IMPORTANT: This creates REAL TFDS datasets that finetune.py can load via dlimp.
Output can be used directly with:
  python finetune.py --data-root /path/to/output --dataset_name libero_safety

Dataset structure (verified):
  19,664 L0 episodes (affordance, human_safety, obstacle_avoidance only)
  3.4M total frames @ 20fps
  Output: TFDS dataset builder with train/validation splits
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import tempfile
import shutil

import numpy as np
import av

# Paths come from fastvla/posttrain/config.env, not from this file. sys.path juggling so
# the script still runs standalone (`python convert_lerobot_to_rlds.py ...`).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load as _load_posttrain_cfg  # noqa: E402

_CFG = _load_posttrain_cfg()
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Set by build_rlds() before the builder is constructed. TFDS instantiates the
# builder itself, so there is no constructor to thread these through.
_BUILD_CTX = {}


class LiberoSafety(tfds.core.GeneratorBasedBuilder):
    """TFDS builder for LIBERO-Safety RLDS dataset.

    The class name matters: TFDS derives the dataset name from it, so this must
    stay `LiberoSafety` for `tfds.builder("libero_safety", data_dir=...)` — the
    call RLDSDataset makes — to resolve.
    """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial LIBERO-Safety L0 release (19,664 episodes)",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Return dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="""
                LIBERO-Safety training dataset: 19,664 L0 episodes across three suites
                (affordance, human_safety, obstacle_avoidance).

                Each episode contains:
                - observation.image: 256×256 RGB frames (third-person)
                - observation.wrist_image: 256×256 RGB frames (wrist camera)
                - observation.state: 8D proprioceptive state
                - action: 7D action (6D pose delta + 1D gripper)
                - language_instruction: task description

                Converted from LeRobot parquet + MP4 format to TFDS.
                """,
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=tf.uint8,
                                        doc="Third-person RGB image"
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=tf.uint8,
                                        doc="Wrist camera RGB image"
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(8,),
                                        dtype=tf.float32,
                                        doc="Proprioceptive state (8D)"
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=tf.float32,
                                doc="Action (6D pose delta + 1D gripper)"
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Task description. Step-level, because "
                                    "make_dataset_from_rlds reads traj[language_key]"
                            ),
                            "is_terminal": tfds.features.Scalar(
                                tf.bool,
                                doc="Is terminal step"
                            ),
                            "is_first": tfds.features.Scalar(
                                tf.bool,
                                doc="Is first step"
                            ),
                            "is_last": tfds.features.Scalar(
                                tf.bool,
                                doc="Is last step"
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "episode_index": tfds.features.Scalar(
                                tf.int32,
                                doc="Episode index (0-19663)"
                            ),
                            "suite": tfds.features.Text(
                                doc="Task suite (affordance, human_safety, obstacle_avoidance)"
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Task description"
                            ),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://libero-safety.github.io",
            citation="""
                @article{cui2026liberosafety,
                  title={LIBERO-Safety: A Comprehensive Benchmark for Physical and Semantic Safety
                         in Vision-Language-Action Models},
                  author={Cui, R. and Zhang, Z. and Pang, J. and others},
                  journal={ECCV 2026},
                  year={2026}
                }
            """,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define train/validation splits from the episode list in _BUILD_CTX."""
        return {
            "train": self._generate_examples(_BUILD_CTX["train"]),
            "validation": self._generate_examples(_BUILD_CTX["validation"]),
        }

    def _generate_examples(self, episode_indices):
        """Yield (key, example) pairs in RLDS layout.

        language_instruction is emitted per step, not in episode_metadata:
        make_dataset_from_rlds pulls it from traj[language_key], so an
        episode-level field is invisible to the loader.
        """
        conv = _BUILD_CTX["converter"]

        for episode_index in episode_indices:
            record = conv.process_episode(int(episode_index))
            if record is None:
                continue

            instruction = record["episode_metadata"]["language_instruction"]
            suite = record["episode_metadata"]["suite"]
            n = len(record["steps"])

            steps = []
            for t, step in enumerate(record["steps"]):
                steps.append({
                    "observation": {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"].astype(np.float32),
                    },
                    "action": step["action"].astype(np.float32),
                    "language_instruction": instruction,
                    "is_first": t == 0,
                    "is_last": t == n - 1,
                    "is_terminal": t == n - 1,
                })

            yield int(episode_index), {
                "steps": steps,
                "episode_metadata": {
                    "episode_index": int(episode_index),
                    "suite": suite,
                    "language_instruction": instruction,
                },
            }


class LiberoSafetyConverter:
    """Main converter class for LIBERO-Safety → TFDS."""

    # Task → (suite, difficulty) mapping
    TASK_MAP = {
        'grab_the_knife_and_cut_the_lemon_on_the_cutting_board': ('affordance', 0),
        'pass_me_the_hammer': ('affordance', 0),
        'pick_up_the_fork_and_place_it_on_the_right_of_the_plate': ('affordance', 0),
        'pour_the_water_in_the_classic_blue_mug_to_the_bowl': ('affordance', 0),
        'put_the_frypan_on_the_stove': ('affordance', 0),
        'pick_the_akita_black_bowl_next_to_the_cookies_box_and_bring_it_for_me': ('human_safety', 0),
        'pick_up_the_banana_and_put_it_on_the_plate_in_my_hand': ('human_safety', 0),
        'pick_up_the_can_of_soda_and_put_it_on_the_plate_in_my_hand': ('human_safety', 0),
        'pick_up_the_red_apple_and_put_it_on_the_plate_in_my_hand': ('human_safety', 0),
        'put_the_banana_on_the_porcelain_plate': ('human_safety', 0),
        'pick_up_the_akita_black_bowl_on_the_stove_and_place_it_on_the_plate': ('obstacle_avoidance', 0),
        'pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy': ('obstacle_avoidance', 0),
        'put_both_moka_pots_on_the_stove': ('obstacle_avoidance', 0),
        'put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket': ('obstacle_avoidance', 0),
        'put_the_white_yellow_mug_in_the_microwave_and_close_it': ('obstacle_avoidance', 0),
    }

    def __init__(self, data_root: Path, output_root: Path, meta_root: Optional[Path] = None):
        """Initialize converter."""
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Determine meta directory
        if meta_root:
            self.meta_root = Path(meta_root)
        elif (self.data_root / "meta").exists():
            self.meta_root = self.data_root / "meta"
        else:
            self.meta_root = Path(_CFG["POSTTRAIN_LEROBOT_META"])

        self.tasks_jsonl = self.meta_root / "tasks.jsonl"
        self.episodes_jsonl = self.meta_root / "episodes.jsonl"
        self.info_json = self.meta_root / "info.json"

        self.tasks_by_index: Dict[int, str] = {}
        self.episodes: List[Dict] = []
        self.info: Dict = {}

        self._load_metadata()

    def _load_metadata(self):
        """Load tasks, episodes, and info metadata."""
        logger.info("Loading metadata...")

        with open(self.tasks_jsonl) as f:
            for line in f:
                obj = json.loads(line)
                self.tasks_by_index[obj['task_index']] = obj['task']

        with open(self.episodes_jsonl) as f:
            for line in f:
                self.episodes.append(json.loads(line))

        with open(self.info_json) as f:
            self.info = json.load(f)

        logger.info(f"Loaded {len(self.tasks_by_index)} tasks")
        logger.info(f"Loaded {len(self.episodes)} episodes")

    def _normalize_task(self, task_str: str) -> str:
        """Normalize task string for matching."""
        return task_str.lower().replace(' ', '_')

    def _map_task_to_suite_difficulty(self, task_description: str) -> Optional[Tuple[str, int]]:
        """Map task description to (suite, difficulty)."""
        normalized = self._normalize_task(task_description)
        if normalized in self.TASK_MAP:
            return self.TASK_MAP[normalized]
        return None

    def _get_episode_chunk(self, episode_index: int) -> int:
        """Get chunk ID for episode index."""
        return episode_index // 1000

    def _load_parquet(self, episode_index: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Load parquet file for episode."""
        chunk_id = self._get_episode_chunk(episode_index)
        parquet_file = (
            self.data_root / "data" / f"chunk-{chunk_id:03d}" /
            f"episode_{episode_index:06d}.parquet"
        )

        df = pd.read_parquet(parquet_file)

        state_list = [np.array(s, dtype=np.float32) for s in df['observation.state']]
        state = np.stack(state_list, axis=0)

        actions_list = [np.array(a, dtype=np.float32) for a in df['actions']]
        actions = np.stack(actions_list, axis=0)

        task_index = int(df['task_index'].iloc[0])

        return state, actions, task_index

    def _decode_video(self, episode_index: int, camera: str) -> np.ndarray:
        """
        Decode MP4 video to frames, in-process, via PyAV (libdav1d).

        These videos are AV1. OpenCV 4.11 opens them and reports a correct frame
        count, then returns False from read() without ever yielding a frame — it
        fails silently rather than raising, so a whole build can look like it
        merely "skipped" episodes. PyAV decodes AV1 and raises on real errors.

        Returns:
            frames [T, 256, 256, 3] uint8, RGB
        """
        chunk_id = self._get_episode_chunk(episode_index)
        video_file = (
            self.data_root / "videos" / f"chunk-{chunk_id:03d}" / camera /
            f"episode_{episode_index:06d}.mp4"
        )

        if not video_file.exists():
            raise FileNotFoundError(f"Video not found: {video_file}")

        frames = []
        with av.open(str(video_file)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))

        if not frames:
            raise ValueError(f"Decoded 0 frames from {video_file}")

        return np.stack(frames, axis=0).astype(np.uint8)

    @staticmethod
    def _check_frame_content(frames: np.ndarray, label: str, episode_index: int):
        """
        Reject frames that have the right shape but no picture in them.

        The shape assertions alone pass just as happily on an array of zeros, which
        is exactly what a broken decode path produces, so content is checked
        separately: no individual frame may be entirely black, the clip mean must
        sit in a plausible range, and the clip must not be a single frozen image.
        """
        flat = frames.reshape(len(frames), -1)

        dead = np.nonzero(flat.max(axis=1) == 0)[0]
        if len(dead):
            raise ValueError(
                f"Episode {episode_index} {label}: {len(dead)} all-zero frame(s), "
                f"first at index {int(dead[0])}"
            )

        mean = float(frames.mean())
        if not (1.0 <= mean <= 254.0):
            raise ValueError(
                f"Episode {episode_index} {label}: implausible mean pixel {mean:.2f}"
            )

        # Temporal, not spatial: a frozen clip has plenty of variance *within* each
        # frame, so a global std is no evidence at all. Compare frames to the first
        # and stop at the first difference, which for real video is frame 1.
        if len(frames) > 1:
            frozen = True
            for f in frames[1:]:
                if (f != frames[0]).any():
                    frozen = False
                    break
            if frozen:
                raise ValueError(
                    f"Episode {episode_index} {label}: every frame identical to frame 0"
                )

        return mean

    def process_episode(self, episode_index: int) -> Optional[Dict]:
        """
        Process single episode to RLDS format (TensorFlow-friendly).

        Returns:
            RLDS episode dict with steps and metadata, or None if skip/error
        """
        try:
            # Get episode metadata
            ep = self.episodes[episode_index]
            task_description = ep['tasks'][0]

            # Map to (suite, difficulty)
            result = self._map_task_to_suite_difficulty(task_description)
            if result is None:
                logger.warning(f"Episode {episode_index}: Could not map task '{task_description}'")
                return None

            suite, difficulty = result

            # Filter to L0 (all episodes are L0 in LeRobot release)
            if difficulty != 0:
                logger.debug(f"Episode {episode_index}: Skipping difficulty={difficulty}")
                return None

            # Load parquet
            state, actions, task_index = self._load_parquet(episode_index)
            T = len(state)

            # Decode videos
            try:
                observation_image = self._decode_video(episode_index, 'observation.image')
                observation_wrist_image = self._decode_video(episode_index, 'observation.wrist_image')
            except FileNotFoundError as e:
                logger.warning(f"Episode {episode_index}: {e}")
                return None

            # Shape alone would pass on an array of zeros, so check the pixels too.
            self._check_frame_content(observation_image, 'observation.image', episode_index)
            self._check_frame_content(observation_wrist_image, 'observation.wrist_image', episode_index)

            # Verify dimensions match
            if observation_image.shape[0] != T or observation_wrist_image.shape[0] != T:
                logger.warning(
                    f"Episode {episode_index}: Frame count mismatch. "
                    f"State={T}, image={observation_image.shape[0]}, wrist={observation_wrist_image.shape[0]}"
                )
                return None

            # Build RLDS episode (steps + metadata)
            steps = []
            for t in range(T):
                step = {
                    'observation': {
                        'image': observation_image[t],
                        'wrist_image': observation_wrist_image[t],
                        'state': state[t],
                    },
                    'action': actions[t],
                    'is_terminal': (t == T - 1),
                    'is_first': (t == 0),
                }
                steps.append(step)

            return {
                'steps': steps,
                'episode_metadata': {
                    'episode_index': episode_index,
                    'suite': suite,
                    'language_instruction': task_description,
                }
            }

        except Exception as e:
            logger.error(f"Episode {episode_index}: {e}", exc_info=True)
            return None

    def write_tfds_examples(self, max_episodes: Optional[int] = None, split: str = "train"):
        """
        Write TFDS examples directly (without using builder).

        Uses tf.data.experimental.save for efficient storage.
        """
        logger.info(f"Writing {split} split...")

        processed_count = 0
        skipped_count = 0
        error_count = 0

        suite_counts = defaultdict(int)
        examples = []

        episodes_to_process = (
            self.episodes[:max_episodes] if max_episodes else self.episodes
        )

        for ep in tqdm(episodes_to_process, desc=f"Processing {split}"):
            episode_index = ep['episode_index']

            try:
                record = self.process_episode(episode_index)

                if record is None:
                    skipped_count += 1
                    continue

                # Create TensorFlow example
                example = self._dict_to_example(record)
                examples.append(example)

                processed_count += 1
                suite_counts[record['episode_metadata']['suite']] += 1

                if processed_count % 100 == 0:
                    logger.info(
                        f"Processed {processed_count} episodes, "
                        f"skipped {skipped_count}, errors {error_count}"
                    )

            except Exception as e:
                logger.error(f"Episode {episode_index}: {e}", exc_info=True)
                error_count += 1

        # Write examples to TFDS format
        split_dir = self.output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Serialize examples to TFRecord format (standard for TFDS)
        record_file = split_dir / "data.tfrecord"
        logger.info(f"Writing {len(examples)} examples to {record_file}...")

        with tf.io.TFRecordWriter(str(record_file)) as writer:
            for example in examples:
                writer.write(example.SerializeToString())

        # Write summary
        summary = {
            "total_episodes": processed_count,
            "skipped": skipped_count,
            "errors": error_count,
            "suite_counts": dict(suite_counts),
            "split": split,
        }

        with open(split_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Wrote {processed_count} episodes to {record_file}")
        logger.info(f"Summary: {summary}")

        return processed_count, suite_counts

    def _dict_to_example(self, record: Dict) -> tf.train.Example:
        """Convert episode dict to TensorFlow Example."""
        steps = record['steps']
        metadata = record['episode_metadata']

        # Serialize steps
        steps_json = json.dumps({
            'images': [step['observation']['image'].tolist() for step in steps],
            'wrist_images': [step['observation']['wrist_image'].tolist() for step in steps],
            'states': [step['observation']['state'].tolist() for step in steps],
            'actions': [step['action'].tolist() for step in steps],
            'is_terminal': [step['is_terminal'] for step in steps],
            'is_first': [step['is_first'] for step in steps],
        })

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'episode_index': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[metadata['episode_index']])
                    ),
                    'suite': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[metadata['suite'].encode('utf-8')])
                    ),
                    'language_instruction': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[metadata['language_instruction'].encode('utf-8')])
                    ),
                    'steps_json': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[steps_json.encode('utf-8')])
                    ),
                }
            )
        )

        return example

    def build_rlds(self, data_dir: Path, max_episodes: Optional[int] = None,
                   val_split: float = 0.1, stratified: bool = False):
        """
        Produce a real TFDS dataset via download_and_prepare().

        The older convert()/write_tfds_examples() path below writes hand-rolled
        TFRecords whose images are JSON integer lists; tfds.builder() cannot open
        that, so RLDSDataset (and therefore finetune.py) cannot read it. This is
        the path that emits something the loader accepts.
        """
        indices = [ep['episode_index'] for ep in self.episodes]

        if max_episodes and stratified:
            # Episodes are grouped by suite in index order, so indices[:N] would be
            # only affordance. An even stride spans all three suites.
            step = max(1, len(indices) // max_episodes)
            indices = indices[::step][:max_episodes]
        elif max_episodes:
            indices = indices[:max_episodes]

        val_count = int(len(indices) * val_split)
        train_idx = indices[:len(indices) - val_count]
        val_idx = indices[len(indices) - val_count:]

        logger.info(f"Building TFDS: {len(train_idx)} train / {len(val_idx)} validation")

        _BUILD_CTX.clear()
        _BUILD_CTX.update({"converter": self, "train": train_idx, "validation": val_idx})

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        builder = LiberoSafety(data_dir=str(data_dir))
        builder.download_and_prepare()

        logger.info(f"TFDS build complete at {data_dir}")
        logger.info(f"  splits: { {k: v.num_examples for k, v in builder.info.splits.items()} }")
        return builder

    def convert(self, max_episodes: Optional[int] = None, val_split: float = 0.1):
        """
        Convert full dataset to TFDS format.

        Args:
            max_episodes: Limit to first N episodes (for testing)
            val_split: Fraction of data to use for validation (0.0-1.0)
        """
        logger.info(f"Converting {len(self.episodes)} episodes to TFDS format...")

        # Split into train/val
        total_ep = len(self.episodes) if max_episodes is None else min(max_episodes, len(self.episodes))
        val_count = int(total_ep * val_split)
        train_count = total_ep - val_count

        logger.info(f"Train episodes: {train_count}")
        logger.info(f"Validation episodes: {val_count}")

        # Temporarily split episodes
        train_episodes = self.episodes[:train_count]
        val_episodes = self.episodes[train_count:train_count+val_count]

        self.episodes = train_episodes
        train_count_out, train_suites = self.write_tfds_examples(
            max_episodes=None, split="train"
        )

        self.episodes = val_episodes
        val_count_out, val_suites = self.write_tfds_examples(
            max_episodes=None, split="validation"
        )

        # Write dataset config
        config = {
            "train_episodes": train_count_out,
            "val_episodes": val_count_out,
            "train_suites": dict(train_suites),
            "val_suites": dict(val_suites),
            "format": "tfrecord",
            "version": "1.0.0",
        }

        with open(self.output_root / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info("\n" + "="*80)
        logger.info("CONVERSION COMPLETE")
        logger.info("="*80)
        logger.info(f"Train: {train_count_out} episodes")
        logger.info(f"Validation: {val_count_out} episodes")
        logger.info(f"Output directory: {self.output_root}")
        logger.info(f"Format: TFDS/TFRecord (ready for finetune.py)")
        logger.info("="*80)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert LIBERO-Safety to TFDS/RLDS format for OFT fine-tuning"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(_CFG["POSTTRAIN_LEROBOT_ROOT"]),
        help="Path to LIBERO-Safety dataset root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(_CFG["POSTTRAIN_RLDS_FULL"]),
        help="Output path for TFDS dataset",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit to first N episodes (for testing)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (0.0-1.0)",
    )

    parser.add_argument(
        "--stratified",
        action="store_true",
        help="With --max-episodes, sample evenly across the index range so the "
             "subset spans all three suites instead of only affordance",
    )
    parser.add_argument(
        "--legacy-tfrecord",
        action="store_true",
        help="Use the old hand-rolled TFRecord writer. Its output is NOT a TFDS "
             "dataset and cannot be read by tfds.load / RLDSDataset.",
    )

    args = parser.parse_args()

    converter = LiberoSafetyConverter(args.data_root, args.output_root)
    if args.legacy_tfrecord:
        converter.convert(max_episodes=args.max_episodes, val_split=args.val_split)
    else:
        converter.build_rlds(
            data_dir=args.output_root,
            max_episodes=args.max_episodes,
            val_split=args.val_split,
            stratified=args.stratified,
        )


if __name__ == "__main__":
    main()
