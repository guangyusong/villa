"""Utility script to sample batches from the dataset pipeline.

Given a config YAML and a data directory (with `images/` and `labels/`), this
script instantiates the corresponding dataset, honours slice-sampling settings,
and iterates a requested number of samples through the PyTorch DataLoader. It
prints per-sample metadata (plane, slice index, sampling weights) so that slice
sampling behaviour can be validated quickly without running a full training
loop.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, List

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Sampler

from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.utilities.data_format_utils import detect_data_format


class FixedOrderSampler(Sampler[int]):
    """Sampler that yields a predetermined sequence of indices."""

    def __init__(self, indices: Iterable[int]):
        self._indices: List[int] = [int(i) for i in indices]

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._indices)


def build_config_manager(config_path: Path, data_path: Path, verbose: bool = True) -> ConfigManager:
    mgr = ConfigManager(verbose=verbose)
    mgr.load_config(str(config_path))

    mgr.data_path = data_path
    detected_format = detect_data_format(data_path)
    if detected_format is None:
        raise ValueError(f"Could not determine data format for {data_path}")
    mgr.data_format = detected_format

    # Force reduced upfront checks for quick sampling.
    mgr.skip_image_checks = True
    mgr.skip_intensity_sampling = True
    mgr.cache_valid_patches = False

    # Mirror onto dataset_config so downstream logic sees the overrides.
    if hasattr(mgr, 'dataset_config') and isinstance(mgr.dataset_config, dict):
        mgr.dataset_config['skip_image_checks'] = True
        mgr.dataset_config['skip_intensity_sampling'] = True
        mgr.dataset_config['cache_valid_patches'] = False

    return mgr


def build_dataset(mgr: ConfigManager, is_training: bool = True):
    fmt = getattr(mgr, 'data_format', 'zarr')
    fmt = fmt.lower()

    if fmt == 'napari':
        from vesuvius.models.datasets.napari_dataset import NapariDataset
        return NapariDataset(mgr=mgr, is_training=is_training)
    if fmt == 'image':
        from vesuvius.models.datasets.image_dataset import ImageDataset
        return ImageDataset(mgr=mgr, is_training=is_training)
    if fmt == 'zarr':
        from vesuvius.models.datasets.zarr_dataset import ZarrDataset
        return ZarrDataset(mgr=mgr, is_training=is_training)
    raise ValueError(f"Unsupported data format '{fmt}'")


def choose_sample_indices(dataset, num_samples: int, seed: int | None):
    rng = np.random.default_rng(seed)
    total = len(dataset)
    if total == 0:
        raise ValueError("Dataset contains zero valid patches")

    weights = getattr(dataset, 'patch_weights', None)
    if isinstance(weights, list) and len(weights) >= total:
        w = np.asarray(weights, dtype=np.float64)
        if not np.any(w > 0):
            w = None
        else:
            w = w / w.sum()
    else:
        w = None

    if w is not None:
        indices = rng.choice(total, size=num_samples, replace=True, p=w)
    else:
        indices = rng.integers(0, total, size=num_samples)

    return indices.tolist(), w


def to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu()
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


def main():
    parser = argparse.ArgumentParser(description="Sample batches from dataset with slice sampling enabled.")
    parser.add_argument('--config', type=Path, required=True, help='Path to training YAML config.')
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory containing images/ and labels/.')
    parser.add_argument('--num-samples', type=int, default=16, help='Number of individual samples to iterate.')
    parser.add_argument('--batch-size', type=int, default=1, help='Loader batch size (default: 1).')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader worker processes (default: 0).')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    parser.add_argument('--output-dir', type=Path, default=None, help='Optional directory to save sampled outputs (images/labels/metadata).')
    parser.add_argument('--save-masks', action='store_true', help='When set, request plane masks from the dataset and save them alongside samples.')

    args = parser.parse_args()

    config_path = args.config.resolve()
    data_path = args.data_dir.resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    mgr = build_config_manager(config_path, data_path)
    if args.save_masks:
        mgr.slice_save_plane_masks = True
        if hasattr(mgr, 'slice_sampling_config') and isinstance(mgr.slice_sampling_config, dict):
            mgr.slice_sampling_config['save_plane_masks'] = True

    dataset = build_dataset(mgr, is_training=True)

    print(f"Dataset created: {dataset.__class__.__name__}")
    print(f" - total valid patches: {len(dataset)}")
    print(f" - slice sampling enabled: {getattr(dataset, 'slice_sampling_enabled', False)}")
    print("Note: reported tilt settings reflect configuration; actual draw per sample is random.")

    indices, weights = choose_sample_indices(dataset, num_samples=args.num_samples, seed=args.seed)
    sampler = FixedOrderSampler(indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    grouped_indices = [
        indices[i:i + args.batch_size]
        for i in range(0, len(indices), args.batch_size)
    ]

    plane_counts: dict[str, int] = {}
    sample_counter = 0

    output_dir = args.output_dir
    if output_dir is not None:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = None
    labels_dir = None
    metadata_dir = None
    masks_dir = None
    extra_dirs = {}
    if output_dir is not None:
        images_dir = output_dir / 'images'
        labels_dir = output_dir / 'labels'
        metadata_dir = output_dir / 'metadata'
        if args.save_masks:
            masks_dir = output_dir / 'masks'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        if masks_dir is not None:
            masks_dir.mkdir(parents=True, exist_ok=True)

    for group_ids, batch in zip(grouped_indices, loader):
        batch_size = len(group_ids)
        images = batch['image']
        assert images.shape[0] == batch_size

        label_keys = [
            key for key, tensor in batch.items()
            if key not in ('image', 'is_unlabeled', 'patch_info', 'regression_keys')
            and isinstance(tensor, torch.Tensor)
        ]
        batch_patch_info = batch.get('patch_info') if 'patch_info' in batch else None
        batch_masks = batch.get('plane_mask') if 'plane_mask' in batch else None

        def materialize(value, offset):
            if isinstance(value, torch.Tensor):
                item = value[offset]
                return item.tolist() if item.ndim else item.item()
            if isinstance(value, (list, tuple)):
                return value[offset]
            if isinstance(value, dict):
                return {k: materialize(v, offset) for k, v in value.items()}
            return value

        def extract_meta(info_source, offset):
            if info_source is None:
                return None
            if isinstance(info_source, (list, tuple, dict, torch.Tensor)):
                return materialize(info_source, offset)
            return info_source

        for offset, idx in enumerate(group_ids):
            patch_info = dataset.valid_patches[idx]
            plane_meta = extract_meta(batch_patch_info, offset)

            plane_value = plane_meta.get('plane') if isinstance(plane_meta, dict) else None
            plane = plane_value if plane_value is not None else patch_info.get('plane', 'n/a')
            plane_counts[plane] = plane_counts.get(plane, 0) + 1

            slice_idx = None
            if isinstance(plane_meta, dict):
                slice_idx = plane_meta.get('slice_index')
            if slice_idx is None:
                slice_idx = patch_info.get('slice_index', 'n/a')
            weight = weights[idx] if weights is not None else None

            tilts_cfg = getattr(dataset, 'slice_random_tilt_planes', {})
            tilt_config = tilts_cfg.get(plane, {})
            actual_angles = None
            if isinstance(plane_meta, dict):
                raw_angles = plane_meta.get('angles')
                if isinstance(raw_angles, dict):
                    actual_angles = {}
                    for axis, val in raw_angles.items():
                        parsed = materialize(val, offset)
                        try:
                            actual_angles[axis] = float(parsed)
                        except (TypeError, ValueError):
                            actual_angles[axis] = parsed

            print(
                f"Sample {sample_counter:03d} | plane={plane} slice={slice_idx} "
                f"weight={weight if weight is not None else 'uniform'}"
            )
            print(
                f"  image tensor shape: {tuple(images[offset].shape)} "
                f"| tilt_cfg={tilt_config if tilt_config else 'none'}"
            )
            if actual_angles:
                angle_parts = []
                for axis, rad in actual_angles.items():
                    if isinstance(rad, (int, float)):
                        if abs(rad) > 1e-6:
                            angle_parts.append(f"{axis}={math.degrees(rad):0.1f}")
                    else:
                        angle_parts.append(f"{axis}={rad}")
                print("  actual angles (deg): " + (", ".join(angle_parts) if angle_parts else "all ≈0"))

            if output_dir is not None:
                base_name = f"sample_{sample_counter:04d}"

                image_array = images[offset].detach().cpu().numpy()
                if image_array.ndim == 3 and image_array.shape[0] == 1:
                    image_to_save = image_array[0]
                else:
                    image_to_save = image_array
                tifffile.imwrite(images_dir / f"{base_name}.tif", image_to_save.astype(np.float32))

                label_files = {}
                extra_files = {}
                for key in label_keys:
                    label_tensor = batch[key][offset]
                    label_array = label_tensor.detach().cpu().numpy()
                    if label_array.ndim == 3 and label_array.shape[0] == 1:
                        label_to_save = label_array[0]
                    else:
                        label_to_save = label_array

                    if 'skel' in key.lower():
                        target_dir = extra_dirs.setdefault(key, output_dir / key)
                        target_dir.mkdir(parents=True, exist_ok=True)
                        filename = f"{base_name}_{key}.tif"
                        tifffile.imwrite(target_dir / filename, label_to_save.astype(np.float32))
                        extra_files.setdefault(key, []).append(filename)
                    else:
                        label_filename = f"{base_name}_{key}.tif"
                        tifffile.imwrite(labels_dir / label_filename, label_to_save.astype(np.float32))
                        label_files[key] = label_filename

                mask_filename = None
                if args.save_masks and batch_masks is not None:
                    if isinstance(batch_masks, torch.Tensor):
                        mask_tensor = batch_masks[offset]
                    else:
                        mask_tensor = materialize(batch_masks, offset)

                    if isinstance(mask_tensor, torch.Tensor):
                        mask_array = mask_tensor.detach().cpu().numpy().astype(np.float32)
                    else:
                        mask_array = np.asarray(mask_tensor, dtype=np.float32)

                    mask_filename = f"{base_name}_mask.tif"
                    tifffile.imwrite((masks_dir / mask_filename), mask_array.astype(np.float32))

                metadata = {
                    'index': int(idx),
                    'plane': plane,
                    'slice_index': slice_idx,
                    'position': plane_meta.get('position') if isinstance(plane_meta, dict) else patch_info.get('position'),
                    'patch_size': plane_meta.get('patch_size') if isinstance(plane_meta, dict) else patch_info.get('patch_size'),
                    'weight': None if weight is None else float(weight),
                    'tilt_config_degrees': tilt_config if tilt_config else {},
                    'angles_rad': actual_angles if actual_angles else {},
                    'image_file': f"{base_name}.tif",
                    'label_files': label_files,
                    'extra_files': extra_files,
                    'mask_file': mask_filename,
                }
                metadata = to_serializable(metadata)
                with open(metadata_dir / f"{base_name}.json", 'w', encoding='utf-8') as meta_f:
                    json.dump(metadata, meta_f, indent=2)

            sample_counter += 1

    print("\nSummary:")
    for plane, count in sorted(plane_counts.items()):
        pct = (count / sample_counter) * 100 if sample_counter else 0
        print(f" - plane {plane}: {count} samples ({pct:.1f}%)")


if __name__ == '__main__':
    main()
