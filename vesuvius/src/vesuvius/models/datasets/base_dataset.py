from pathlib import Path
import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
import fsspec
import zarr
from torch.utils.data import Dataset
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from vesuvius.models.training.auxiliary_tasks import (
    compute_distance_transform,
    compute_surface_normals_from_sdt,
    compute_structure_tensor,
    compute_inplane_direction,
)
from vesuvius.models.augmentation.transforms.spatial.transpose import TransposeAxesTransform
# Augmentations will be handled directly in this file
from vesuvius.models.augmentation.transforms.utils.random import RandomTransform
from vesuvius.models.augmentation.helpers.scalar_type import RandomScalar
from vesuvius.models.augmentation.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from vesuvius.models.augmentation.transforms.intensity.contrast import ContrastTransform, BGContrast
from vesuvius.models.augmentation.transforms.intensity.gamma import GammaTransform
from vesuvius.models.augmentation.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from vesuvius.models.augmentation.transforms.noise.gaussian_blur import GaussianBlurTransform
from vesuvius.models.augmentation.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from vesuvius.models.augmentation.transforms.spatial.mirroring import MirrorTransform
from vesuvius.models.augmentation.transforms.spatial.spatial import SpatialTransform
from vesuvius.models.augmentation.transforms.utils.compose import ComposeTransforms
from vesuvius.models.augmentation.transforms.noise.extranoisetransforms import BlankRectangleTransform, RicianNoiseTransform, SmearTransform
from vesuvius.models.augmentation.transforms.intensity.illumination import InhomogeneousSliceIlluminationTransform

from vesuvius.utils.utils import pad_or_crop_3d, pad_or_crop_2d
from ..training.normalization import get_normalization
from .intensity_properties import initialize_intensity_properties
from .find_valid_patches import find_valid_patches, compute_bounding_box_3d, bounding_box_volume
from .save_valid_patches import save_valid_patches, load_cached_patches

class BaseDataset(Dataset):
    """
    A PyTorch Dataset base class for handling both 2D and 3D data from various sources.
    
    Subclasses must implement the _initialize_volumes() method to specify how
    data is loaded from their specific data source.
    """
    def __init__(self,
                 mgr,
                 is_training=True):
        """
        Initialize the dataset with configuration from the manager.
        
        Parameters
        ----------
        mgr : ConfigManager
            Manager containing configuration parameters
        is_training : bool
            Whether this dataset is for training (applies augmentations) or validation

        """
        super().__init__()
        self.mgr = mgr
        self.is_training = is_training

        self.model_name = mgr.model_name
        self.targets = mgr.targets               # e.g. {"ink": {...}, "normals": {...}}
        self.patch_size = mgr.train_patch_size   # Expected to be [z, y, x]
        self.min_labeled_ratio = mgr.min_labeled_ratio
        self.min_bbox_percent = mgr.min_bbox_percent
        
        # if you are certain your data contains dense labels (everything is labeled), you can choose
        # to skip the valid patch finding
        self.skip_patch_validation = getattr(mgr, 'skip_patch_validation', False)

        # for semi-supervised workflows, unlabeled data is obviously needed,
        # we want a flag for this so in fully supervised workflows we can assert that all images have
        # corresponding labels (so we catch it early)
        self.allow_unlabeled_data = getattr(mgr, 'allow_unlabeled_data', False)

        # Initialize normalization (will be set after computing intensity properties)
        self.normalization_scheme = getattr(mgr, 'normalization_scheme', 'zscore')
        self.intensity_properties = getattr(mgr, 'intensity_properties', {})
        self.normalizer = None  # Will be initialized after volumes are loaded

        self.target_volumes = {}
        self.valid_patches = []
        self.patch_weights = None
        self.is_2d_dataset = None
        self.data_path = Path(mgr.data_path) if hasattr(mgr, 'data_path') else None
        self.zarr_arrays = []
        self.zarr_names = []
        self.data_paths = []

        # Slice sampling configuration (2D slices from 3D inputs)
        self.slice_sampling_enabled = bool(getattr(mgr, 'slice_sampling_enabled', False))
        self.slice_sample_planes = list(getattr(mgr, 'slice_sample_planes', []))
        self.slice_plane_weights = dict(getattr(mgr, 'slice_plane_weights', {}))
        self.slice_plane_patch_sizes = dict(getattr(mgr, 'slice_plane_patch_sizes', {}))
        self.slice_primary_plane = getattr(mgr, 'slice_primary_plane', None)
        self.slice_random_rotation_planes = dict(getattr(mgr, 'slice_random_rotation_planes', {}))
        self.slice_random_tilt_planes = dict(getattr(mgr, 'slice_random_tilt_planes', {}))
        self.slice_label_interpolation = getattr(mgr, 'slice_label_interpolation', {})
        self.slice_save_plane_masks = bool(getattr(mgr, 'slice_save_plane_masks', False))
        self.slice_plane_mask_mode = getattr(mgr, 'slice_plane_mask_mode', 'plane')
        if self.slice_plane_mask_mode not in ('volume', 'plane'):
            self.slice_plane_mask_mode = 'plane'

        self.slice_mask_volume_shape = None

        if self.slice_sampling_enabled and not self.slice_sample_planes:
            self.slice_sample_planes = ['z']
            self.slice_plane_weights = {'z': 1.0}

        # Disable cached patch lookup in slice mode (cache incompatible with plane metadata)
        if self.slice_sampling_enabled and getattr(mgr, 'cache_valid_patches', True):
            print("Slice sampling mode: disabling cache_valid_patches (not supported)")
            self.cache_enabled = False
        else:
            self.cache_enabled = getattr(mgr, 'cache_valid_patches', True)

        self.cache_dir = None
        if self.data_path is not None:
            self.cache_dir = self.data_path / '.patches_cache'
            print(f"Cache directory: {self.cache_dir}")
            print(f"Cache enabled: {self.cache_enabled}")

        self._initialize_volumes()

        if self.slice_sampling_enabled and self.slice_save_plane_masks and self.slice_plane_mask_mode == 'volume':
            self._initialize_slice_mask_volume_shape()

        ref_target = list(self.target_volumes.keys())[0]
        ref_volume_data = self.target_volumes[ref_target][0]['data']

        if 'label' in ref_volume_data and ref_volume_data['label'] is not None:
            ref_shape = ref_volume_data['label'].shape
        else:
            ref_shape = ref_volume_data['data'].shape

        # Allow explicit override from config to avoid misclassification.
        force_2d = getattr(self.mgr, 'force_2d', False)
        if self.slice_sampling_enabled:
            force_2d = True
        if not force_2d and hasattr(self.mgr, 'dataset_config'):
            force_2d = bool(self.mgr.dataset_config.get('force_2d', False))
        force_3d = getattr(self.mgr, 'force_3d', False)
        if self.slice_sampling_enabled:
            force_3d = False
        if not force_3d and hasattr(self.mgr, 'dataset_config'):
            force_3d = bool(self.mgr.dataset_config.get('force_3d', False))

        if force_2d and force_3d:
            raise ValueError("Both force_2d and force_3d are set; choose only one.")

        if force_2d:
            self.is_2d_dataset = True
        elif force_3d:
            self.is_2d_dataset = False
        else:
            # Only treat as 2D when the data truly has 2 dimensions
            self.is_2d_dataset = (len(ref_shape) == 2)
        
        if self.is_2d_dataset:
            print("Detected 2D dataset")
            if len(self.patch_size) == 3:
                self.patch_size = list(self.patch_size[-2:])
                print(f"Adjusted patch size for 2D data: {self.patch_size}")
        else:
            print("Detected 3D dataset")

        # Check if we should skip intensity sampling
        skip_intensity_sampling = getattr(mgr, 'skip_intensity_sampling', False)

        if skip_intensity_sampling:
            print("Skipping intensity sampling as requested")
            # Use default values if intensity properties not provided
            if not self.intensity_properties:
                self.intensity_properties = {
                    'mean': 0.0,
                    'std': 1.0,
                    'min': 0.0,
                    'max': 1.0,
                    'percentile_00_5': 0.0,
                    'percentile_99_5': 1.0
                }
        else:
            self.intensity_properties = initialize_intensity_properties(
                target_volumes=self.target_volumes,
                normalization_scheme=self.normalization_scheme,
                existing_properties=self.intensity_properties,
                cache_enabled=self.cache_enabled,
                cache_dir=self.cache_dir,
                mgr=self.mgr,
                sample_ratio=0.001,
                max_samples=1000000
            )

        self.normalizer = get_normalization(self.normalization_scheme, self.intensity_properties)

        self.transforms = None
        if self.is_training:
            self.transforms = self._create_training_transforms()
            print("Training transforms initialized")
        else:
            # For validation, we might still need skeleton transform
            self.transforms = self._create_validation_transforms()
            if self.transforms is not None:
                print("Validation transforms initialized")

        if not self.skip_patch_validation:
            self._get_valid_patches()
        else:
            print("Skipping patch validation as requested")
            if self.slice_sampling_enabled:
                self._generate_all_slice_patches()
            else:
                # Generate all possible patches without validation
                self.valid_patches = []

                # Get the first target to access all volumes (since all targets share the same volumes)
                first_target = list(self.target_volumes.keys())[0]
                all_volumes = self.target_volumes[first_target]

                for vol_idx, volume_info in enumerate(all_volumes):
                    # Get the image array (data) for this volume
                    img_array = volume_info['data']['data']
                    vol_name = volume_info.get('volume_id', f"volume_{vol_idx}")

                    positions = self._get_all_sliding_window_positions(
                        volume_shape=img_array.shape,
                        patch_size=self.patch_size
                    )
                    for pos_dict in positions:
                        self.valid_patches.append({
                            "volume_index": vol_idx,
                            "volume_name": vol_name,
                            "position": pos_dict['start_pos']
                        })
                print(f"Generated {len(self.valid_patches)} patches without validation")

    def _initialize_volumes(self):
        """
        Initialize volumes from the data source.
        
        This method must be implemented by subclasses to specify how
        data is loaded from their specific data source (napari, TIFs, Zarr, etc.).
        
        The implementation should:
        1. Populate self.target_volumes in the format:
           {
               'target_name': [
                   {
                       'data': {
                           'data': numpy_array,      # Image data
                           'label': numpy_array      # Label data
                       }
                   },
                   ...  # Additional volumes for this target
               ],
               ...  # Additional targets
           }

        2. Populate zarr arrays for patch finding:
           - self.zarr_arrays: List of zarr arrays (label volumes)
           - self.zarr_names: List of names for each volume
           - self.data_paths: List of data paths for each volume
        """
        raise NotImplementedError("Subclasses must implement _initialize_volumes() method")

    def _needs_skeleton_transform(self):
        """
        Check if any configured loss requires skeleton data.
        
        Returns
        -------
        bool
            True if skeleton transform should be added to the pipeline
        """
        skeleton_losses = ['MedialSurfaceRecall', 'DC_SkelREC_and_CE_loss', 'SoftSkeletonRecallLoss']
        
        for target_name, target_info in self.targets.items():
            if "losses" in target_info:
                for loss_cfg in target_info["losses"]:
                    if loss_cfg["name"] in skeleton_losses:
                        return True
        return False

    def _get_all_sliding_window_positions(self, volume_shape, patch_size, stride=None):
        """
        Generate all possible sliding window positions for a volume.
        
        Parameters
        ----------
        volume_shape : tuple
            Shape of the volume (2D or 3D)
        patch_size : tuple
            Size of patches to extract
        stride : tuple, optional
            Stride for sliding window, defaults to no overlap (stride = patch_size)
            
        Returns
        -------
        list
            List of positions as dictionaries with 'start_pos' key
        """
        if len(volume_shape) == 2:
            # 2D case
            H, W = volume_shape
            h, w = patch_size
            
            if stride is None:
                stride = (h, w)  # No overlap by default
            
            positions = []
            
            # Generate regular grid positions
            y_positions = list(range(0, H - h + 1, stride[0]))
            x_positions = list(range(0, W - w + 1, stride[1]))
            
            # Check if we need edge patches to cover the entire volume
            # Add final position at bottom edge if there's remaining space
            if y_positions and y_positions[-1] + h < H:
                y_positions.append(H - h)
            
            # Add final position at right edge if there's remaining space  
            if x_positions and x_positions[-1] + w < W:
                x_positions.append(W - w)
            
            total_positions = len(y_positions) * len(x_positions)
            
            with tqdm(total=total_positions, desc="Generating 2D sliding window positions", leave=False) as pbar:
                for y in y_positions:
                    for x in x_positions:
                        positions.append({
                            'start_pos': [y, x]  # 2D uses [y, x]
                        })
                        pbar.update(1)
                
        else:
            # 3D case
            D, H, W = volume_shape
            d, h, w = patch_size
            
            if stride is None:
                stride = (d, h, w)  # No overlap by default
            
            positions = []
            
            # Generate regular grid positions
            z_positions = list(range(0, D - d + 1, stride[0]))
            y_positions = list(range(0, H - h + 1, stride[1]))
            x_positions = list(range(0, W - w + 1, stride[2]))
            
            # Check if we need edge patches to cover the entire volume
            # Add final position at depth edge if there's remaining space
            if z_positions and z_positions[-1] + d < D:
                z_positions.append(D - d)
            
            # Add final position at bottom edge if there's remaining space
            if y_positions and y_positions[-1] + h < H:
                y_positions.append(H - h)
            
            # Add final position at right edge if there's remaining space
            if x_positions and x_positions[-1] + w < W:
                x_positions.append(W - w)
            
            total_positions = len(z_positions) * len(y_positions) * len(x_positions)
            
            with tqdm(total=total_positions, desc="Generating 3D sliding window positions", leave=False) as pbar:
                for z in z_positions:
                    for y in y_positions:
                        for x in x_positions:
                            positions.append({
                                'start_pos': [z, y, x]
                            })
                            pbar.update(1)

        seen = set()
        unique_positions = []
        for pos in positions:
            pos_tuple = tuple(pos['start_pos'])
            if pos_tuple not in seen:
                seen.add(pos_tuple)
                unique_positions.append(pos)
        
        return unique_positions
    
    def _get_valid_patches(self):
        """Find valid patches based on labeled ratio requirements."""
        if self.slice_sampling_enabled:
            self._get_valid_slice_patches()
            return

        # Check if we should load approved patches from vc_proofreader
        if hasattr(self.mgr, 'approved_patches_file') and self.mgr.approved_patches_file:
            if self._load_approved_patches():
                return

        bbox_threshold = getattr(self.mgr, 'min_bbox_percent', 0.97)
        downsample_level = getattr(self.mgr, 'downsample_level', 1)
        num_workers = getattr(self.mgr, 'num_workers', 8)

        if self.cache_enabled and len(self.zarr_arrays) > 0:
            cached_patches = load_cached_patches(
                train_data_paths=self.data_paths,
                label_paths=self.data_paths,
                patch_size=tuple(self.patch_size),
                min_labeled_ratio=self.min_labeled_ratio,
                bbox_threshold=bbox_threshold,
                downsample_level=downsample_level,
                cache_path=str(self.cache_dir) if self.cache_dir else None
            )

            if cached_patches is not None:
                self.valid_patches = cached_patches
                print(f"Successfully loaded {len(self.valid_patches)} patches from cache\n")
                return

        if len(self.zarr_arrays) == 0:
            raise ValueError("No zarr arrays available for patch finding. Subclasses must populate self.zarr_arrays")

        print("Computing valid patches using zarr arrays...")
        valid_patches = find_valid_patches(
            label_arrays=self.zarr_arrays,
            label_names=self.zarr_names,
            patch_size=tuple(self.patch_size),
            bbox_threshold=bbox_threshold,
            label_threshold=self.min_labeled_ratio,
            num_workers=num_workers,
            downsample_level=downsample_level
        )

        # Convert to the expected format
        self.valid_patches = []
        for patch in valid_patches:
            self.valid_patches.append({
                "volume_index": patch["volume_idx"],
                "volume_name": patch["volume_name"],
                "position": patch["start_pos"]  # (z,y,x) for 3D or (y,x) for 2D
            })

        # Save to cache after computing
        if self.cache_enabled and self.cache_dir is not None and len(self.zarr_arrays) > 0:
            cache_path = save_valid_patches(
                valid_patches=valid_patches,
                train_data_paths=self.data_paths,
                label_paths=self.data_paths,
                patch_size=tuple(self.patch_size),
                min_labeled_ratio=self.min_labeled_ratio,
                bbox_threshold=bbox_threshold,
                downsample_level=downsample_level,
                cache_path=str(self.cache_dir) if self.cache_dir else None
            )
            print(f"Saved patches to cache: {cache_path}")

    def _get_valid_slice_patches(self):
        """Generate valid slice-based patches across configured planes."""
        plane_patches = {axis: [] for axis in self.slice_sample_planes}

        first_target = list(self.target_volumes.keys())[0]
        volumes = self.target_volumes[first_target]

        for vol_idx, volume_info in enumerate(volumes):
            data_arr = volume_info['data']['data']
            label_arr = volume_info['data'].get('label')
            volume_name = volume_info.get('volume_id', f"volume_{vol_idx}")

            collected = self._collect_slice_patches_for_volume(
                vol_idx=vol_idx,
                volume_name=volume_name,
                data_arr=data_arr,
                label_arr=label_arr,
                validate=True
            )

            for axis, patches in collected.items():
                if patches:
                    plane_patches.setdefault(axis, []).extend(patches)

        self._finalize_slice_patch_lists(plane_patches)
        print(f"Found {len(self.valid_patches)} valid slice patches across {len(self.target_volumes[first_target])} volume(s)")

    def _generate_all_slice_patches(self):
        """Generate slice patches without label-based validation."""
        plane_patches = {axis: [] for axis in self.slice_sample_planes}

        first_target = list(self.target_volumes.keys())[0]
        volumes = self.target_volumes[first_target]

        for vol_idx, volume_info in enumerate(volumes):
            data_arr = volume_info['data']['data']
            label_arr = volume_info['data'].get('label')
            volume_name = volume_info.get('volume_id', f"volume_{vol_idx}")

            collected = self._collect_slice_patches_for_volume(
                vol_idx=vol_idx,
                volume_name=volume_name,
                data_arr=data_arr,
                label_arr=label_arr,
                validate=False
            )

            for axis, patches in collected.items():
                if patches:
                    plane_patches.setdefault(axis, []).extend(patches)

        self._finalize_slice_patch_lists(plane_patches)
        print(f"Generated {len(self.valid_patches)} slice patches (validation skipped)")

    def _finalize_slice_patch_lists(self, plane_patches):
        """Flatten plane-specific patch lists and compute sampling weights."""
        available_axes = {axis: patches for axis, patches in plane_patches.items() if patches}

        if not available_axes:
            raise ValueError("Slice sampling mode did not find any candidate patches. Check labels and configuration.")

        self.valid_patches = []
        axis_counts = {}

        for axis in self.slice_sample_planes:
            patches = available_axes.get(axis, [])
            if not patches:
                if self.slice_plane_weights.get(axis, 0) > 0:
                    print(f"Warning: no patches produced for plane '{axis}' despite positive sample rate")
                continue
            for patch in patches:
                self.valid_patches.append(patch)
                axis_counts[axis] = axis_counts.get(axis, 0) + 1

        if not self.valid_patches:
            raise ValueError("Slice sampling produced zero patches after filtering. Adjust sample_planes or labels.")

        self.patch_weights = []
        for patch in self.valid_patches:
            axis = patch['plane']
            axis_weight = float(self.slice_plane_weights.get(axis, 1.0))
            count = axis_counts.get(axis, 0)
            if count <= 0 or axis_weight <= 0:
                weight_per_patch = 0.0
            else:
                weight_per_patch = axis_weight / count
            self.patch_weights.append(weight_per_patch)

    def _collect_slice_patches_for_volume(self, vol_idx, volume_name, data_arr, label_arr, validate):
        plane_results = {axis: [] for axis in self.slice_sample_planes}

        spatial_shape = self._extract_spatial_shape(data_arr)
        if spatial_shape is None or len(spatial_shape) != 3:
            print(f"Warning: unable to determine spatial shape for volume '{volume_name}'")
            return plane_results

        perform_validation = bool(validate and label_arr is not None)

        if validate and label_arr is None and not self.allow_unlabeled_data:
            print(f"Skipping unlabeled volume '{volume_name}' (allow_unlabeled_data=False)")
            return plane_results

        for axis in self.slice_sample_planes:
            patch_size = tuple(int(v) for v in self.slice_plane_patch_sizes.get(axis, self.patch_size))
            if len(patch_size) != 2:
                raise ValueError(f"Slice patch size for plane '{axis}' must have 2 elements; got {patch_size}")

            plane_shape = self._plane_shape_from_spatial(spatial_shape, axis)
            if plane_shape is None:
                continue

            positions = self._iter_plane_positions(plane_shape, patch_size)
            if not positions:
                continue

            axis_index = {'z': 0, 'y': 1, 'x': 2}[axis]
            num_slices = spatial_shape[axis_index]

            for slice_idx in range(num_slices):
                mask_slice = None
                if perform_validation:
                    mask_slice = self._extract_label_slice_mask(label_arr, axis, slice_idx)
                    if mask_slice is None or not mask_slice.any():
                        continue

                for pos0, pos1 in positions:
                    if perform_validation and mask_slice is not None:
                        mask_patch = mask_slice[pos0:pos0 + patch_size[0], pos1:pos1 + patch_size[1]]
                        if mask_patch.shape != tuple(patch_size):
                            mask_patch = pad_or_crop_2d(mask_patch, patch_size)
                        if not self._mask_satisfies_thresholds(mask_patch):
                            continue

                    plane_results[axis].append({
                        "volume_index": vol_idx,
                        "volume_name": volume_name,
                        "plane": axis,
                        "slice_index": slice_idx,
                        "position": [int(pos0), int(pos1)],
                        "patch_size": [int(patch_size[0]), int(patch_size[1])]
                    })

        return plane_results

    def _extract_spatial_shape(self, array):
        if array is None:
            return None
        shape = array.shape
        if len(shape) == 4:  # (C, Z, Y, X)
            return shape[1:]
        if len(shape) == 3:  # (Z, Y, X)
            return shape
        if len(shape) == 2:  # (Y, X) -> treat depth of 1
            return (1, shape[0], shape[1])
        return None

    def _plane_shape_from_spatial(self, spatial_shape, axis):
        if spatial_shape is None or len(spatial_shape) != 3:
            return None
        z, y, x = spatial_shape
        if axis == 'z':
            return (y, x)
        if axis == 'y':
            return (z, x)
        if axis == 'x':
            return (z, y)
        return None

    def _iter_plane_positions(self, plane_shape, patch_size):
        if plane_shape is None:
            return []
        h, w = plane_shape
        ph, pw = patch_size

        if ph <= 0 or pw <= 0 or h <= 0 or w <= 0:
            return []

        if ph >= h:
            y_positions = [0]
        else:
            y_positions = list(range(0, h - ph + 1, ph))
            if y_positions[-1] + ph < h:
                y_positions.append(h - ph)

        if pw >= w:
            x_positions = [0]
        else:
            x_positions = list(range(0, w - pw + 1, pw))
            if x_positions[-1] + pw < w:
                x_positions.append(w - pw)

        positions = []
        for y in y_positions:
            for x in x_positions:
                positions.append((int(y), int(x)))
        return positions

    def _extract_label_slice_mask(self, label_arr, axis, slice_idx):
        try:
            if label_arr is None:
                return None
            if label_arr.ndim == 4:
                # (C, Z, Y, X)
                if axis == 'z':
                    slice_data = label_arr[:, slice_idx, :, :]
                elif axis == 'y':
                    slice_data = label_arr[:, :, slice_idx, :]
                else:  # axis == 'x'
                    slice_data = label_arr[:, :, :, slice_idx]
                mask = np.any(slice_data > 0, axis=0)
            elif label_arr.ndim == 3:
                if axis == 'z':
                    slice_data = label_arr[slice_idx, :, :]
                elif axis == 'y':
                    slice_data = label_arr[:, slice_idx, :]
                else:
                    slice_data = label_arr[:, :, slice_idx]
                mask = slice_data > 0
            elif label_arr.ndim == 2:
                mask = label_arr > 0
            else:
                return None
            return np.asarray(mask, dtype=bool)
        except Exception as exc:
            print(f"Warning: failed to extract label slice at plane '{axis}' index {slice_idx}: {exc}")
            return None

    def _mask_satisfies_thresholds(self, mask_patch):
        if mask_patch is None:
            return True
        if not np.any(mask_patch):
            return False

        bbox = compute_bounding_box_3d(mask_patch)
        if bbox is None:
            return False

        bb_vol = bounding_box_volume(bbox)
        patch_vol = mask_patch.size

        if patch_vol == 0:
            return False

        if (bb_vol / patch_vol) < self.min_bbox_percent:
            return False

        labeled_ratio = np.count_nonzero(mask_patch) / patch_vol
        return labeled_ratio >= self.min_labeled_ratio

    def _sample_yaw_angle(self, plane):
        cfg = self.slice_random_rotation_planes.get(plane)
        if not cfg:
            return 0.0
        probability = 1.0
        if isinstance(cfg, dict):
            max_deg = float(cfg.get('max_degrees', 0.0))
            probability = float(cfg.get('probability', 1.0))
        else:
            try:
                max_deg = float(cfg)
            except (TypeError, ValueError):
                return 0.0
        if max_deg <= 0:
            return 0.0
        if probability < 1.0 and np.random.random() > probability:
            return 0.0
        angle_deg = np.random.uniform(-max_deg / 2.0, max_deg / 2.0)
        return math.radians(angle_deg)

    def _sample_tilt_angles(self, plane):
        cfg = self.slice_random_tilt_planes.get(plane, {})
        probability = 1.0
        axis_cfg = cfg
        if isinstance(cfg, dict) and ('axes' in cfg or 'probability' in cfg):
            probability = float(cfg.get('probability', 1.0))
            axis_cfg = cfg.get('axes', {})

        if probability < 1.0 and np.random.random() > probability:
            return {}

        angles = {}
        if isinstance(axis_cfg, dict):
            for axis_key, max_deg in axis_cfg.items():
                try:
                    max_deg = float(max_deg)
                except (TypeError, ValueError):
                    continue
                if max_deg <= 0:
                    continue
                angle_deg = np.random.uniform(-max_deg / 2.0, max_deg / 2.0)
                angles[axis_key] = math.radians(angle_deg)
        return angles

    @staticmethod
    def _rotation_matrix(rx, ry, rz):
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)

        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx]
        ], dtype=np.float32)

        Ry = np.array([
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy]
        ], dtype=np.float32)

        Rz = np.array([
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        return Rz @ Ry @ Rx

    @staticmethod
    def _ensure_perpendicular(vector):
        candidate = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(vector, candidate)) > 0.9:
            candidate = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        perp = np.cross(vector, candidate)
        norm = np.linalg.norm(perp)
        if norm < 1e-6:
            candidate = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            perp = np.cross(vector, candidate)
            norm = np.linalg.norm(perp)
        return perp / max(norm, 1e-8)

    def _compute_plane_center(self, plane, slice_idx, pos0, pos1, patch_size):
        ph, pw = patch_size
        half_u = (ph - 1) / 2.0
        half_v = (pw - 1) / 2.0

        if plane == 'z':
            return np.array([
                pos1 + half_v,
                pos0 + half_u,
                float(slice_idx)
            ], dtype=np.float32)
        if plane == 'y':
            return np.array([
                pos1 + half_v,
                float(slice_idx),
                pos0 + half_u
            ], dtype=np.float32)
        # plane == 'x'
        return np.array([
            float(slice_idx),
            pos1 + half_v,
            pos0 + half_u
        ], dtype=np.float32)

    def _compute_plane_orientation(self, plane, slice_idx, pos0, pos1, patch_size, yaw_angle, tilt_angles):
        base_normals = {
            'z': np.array([0.0, 0.0, 1.0], dtype=np.float32),
            'y': np.array([0.0, 1.0, 0.0], dtype=np.float32),
            'x': np.array([1.0, 0.0, 0.0], dtype=np.float32)
        }
        base_u = {
            'z': np.array([0.0, 1.0, 0.0], dtype=np.float32),
            'y': np.array([0.0, 0.0, 1.0], dtype=np.float32),
            'x': np.array([0.0, 0.0, 1.0], dtype=np.float32)
        }
        base_v = {
            'z': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'y': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'x': np.array([0.0, 1.0, 0.0], dtype=np.float32)
        }

        center = self._compute_plane_center(plane, slice_idx, pos0, pos1, patch_size)

        rx = tilt_angles.get('x', 0.0)
        ry = tilt_angles.get('y', 0.0)
        rz = yaw_angle + tilt_angles.get('z', 0.0)

        rot = self._rotation_matrix(rx, ry, rz)

        u_dir = rot @ base_u[plane]
        v_dir = rot @ base_v[plane]

        u_norm = np.linalg.norm(u_dir)
        if u_norm < 1e-6:
            u_dir = base_u[plane]
        else:
            u_dir = u_dir / u_norm

        v_dir = v_dir - np.dot(v_dir, u_dir) * u_dir
        v_norm = np.linalg.norm(v_dir)
        if v_norm < 1e-6:
            v_dir = self._ensure_perpendicular(u_dir)
        else:
            v_dir = v_dir / v_norm

        normal = np.cross(u_dir, v_dir)
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-6:
            normal = base_normals[plane]
        else:
            normal = normal / n_norm

        return {
            'center': center,
            'u_dir': u_dir,
            'v_dir': v_dir,
            'normal': normal
        }

    def _slice_array_patch(self, array, axis, slice_idx, pos0, pos1, patch_size):
        if array is None:
            return None

        ph, pw = patch_size

        try:
            if array.ndim == 4:  # (C, Z, Y, X)
                if axis == 'z':
                    patch = array[:, slice_idx, pos0:pos0 + ph, pos1:pos1 + pw]
                elif axis == 'y':
                    patch = array[:, pos0:pos0 + ph, slice_idx, pos1:pos1 + pw]
                else:  # axis == 'x'
                    patch = array[:, pos0:pos0 + ph, pos1:pos1 + pw, slice_idx]

                if patch.ndim != 3:
                    patch = np.asarray(patch)
                channels = patch.shape[0]
                padded = [pad_or_crop_2d(patch[c], patch_size) for c in range(channels)]
                return np.stack(padded, axis=0)

            else:
                arr = array
                if arr.ndim == 2:
                    arr = np.expand_dims(arr, axis=0)

                if axis == 'z':
                    patch = arr[slice_idx, pos0:pos0 + ph, pos1:pos1 + pw]
                elif axis == 'y':
                    patch = arr[pos0:pos0 + ph, slice_idx, pos1:pos1 + pw]
                else:
                    patch = arr[pos0:pos0 + ph, pos1:pos1 + pw, slice_idx]

                return pad_or_crop_2d(patch, patch_size)

        except Exception as exc:
            print(f"Warning: failed to slice array on plane '{axis}' index {slice_idx}: {exc}")

        return np.zeros(patch_size, dtype=np.float32)

    def _sample_rotated_plane(self, array, plane, slice_idx, pos0, pos1, patch_size, orientation,
                              interpolation='linear', return_mask=False):
        if array is None or orientation is None:
            return (None, (None, None))

        has_channels = False

        if array.ndim == 4:
            has_channels = True
            _, depth_z, depth_y, depth_x = array.shape
            full_shape = array.shape[1:]
        elif array.ndim == 3:
            depth_z, depth_y, depth_x = array.shape
            full_shape = array.shape
        elif array.ndim == 2:
            depth_z, depth_y, depth_x = 1, array.shape[0], array.shape[1]
            array = array[np.newaxis, ...]
            full_shape = (depth_z, depth_y, depth_x)
        else:
            return (None, (None, None))

        ph, pw = patch_size
        half_u = (ph - 1) / 2.0
        half_v = (pw - 1) / 2.0

        center = orientation['center']
        u_dir = orientation['u_dir']
        v_dir = orientation['v_dir']

        corners = []
        for su in (-half_u, half_u):
            for sv in (-half_v, half_v):
                corners.append(center + su * u_dir + sv * v_dir)
        corners = np.stack(corners, axis=0)

        x_coords = corners[:, 0]
        y_coords = corners[:, 1]
        z_coords = corners[:, 2]

        x_min = max(0, int(math.floor(np.min(x_coords))) - 2)
        x_max = min(depth_x - 1, int(math.ceil(np.max(x_coords))) + 2)
        y_min = max(0, int(math.floor(np.min(y_coords))) - 2)
        y_max = min(depth_y - 1, int(math.ceil(np.max(y_coords))) + 2)
        z_min = max(0, int(math.floor(np.min(z_coords))) - 2)
        z_max = min(depth_z - 1, int(math.ceil(np.max(z_coords))) + 2)

        if z_min > z_max or y_min > y_max or x_min > x_max:
            return (None, (None, None))

        if has_channels:
            sub_volume = np.asarray(array[:, z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1], dtype=np.float32)
            depth = sub_volume.shape[1]
            height = sub_volume.shape[2]
            width = sub_volume.shape[3]
        else:
            sub_volume = np.asarray(array[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1], dtype=np.float32)
            depth = sub_volume.shape[0]
            height = sub_volume.shape[1]
            width = sub_volume.shape[2]

        if depth <= 0 or height <= 0 or width <= 0:
            return (None, (None, None))

        center_local = np.array([
            center[0] - x_min,
            center[1] - y_min,
            center[2] - z_min
        ], dtype=np.float32)

        grid_u = np.arange(ph, dtype=np.float32) - half_u
        grid_v = np.arange(pw, dtype=np.float32) - half_v
        uu, vv = np.meshgrid(grid_u, grid_v, indexing='ij')

        coord_local = center_local[np.newaxis, np.newaxis, :] + uu[..., np.newaxis] * u_dir[np.newaxis, np.newaxis, :] + vv[..., np.newaxis] * v_dir[np.newaxis, np.newaxis, :]
        coord_global = center[np.newaxis, np.newaxis, :] + uu[..., np.newaxis] * u_dir[np.newaxis, np.newaxis, :] + vv[..., np.newaxis] * v_dir[np.newaxis, np.newaxis, :]

        coords_x = coord_local[..., 0]
        coords_y = coord_local[..., 1]
        coords_z = coord_local[..., 2]

        def normalize(coords, size):
            if size <= 1:
                return np.zeros_like(coords, dtype=np.float32)
            return (2.0 * coords / (size - 1)) - 1.0

        x_norm = normalize(coords_x, width)
        y_norm = normalize(coords_y, height)
        z_norm = normalize(coords_z, depth)

        grid = np.stack([x_norm, y_norm, z_norm], axis=-1).astype(np.float32)
        grid = grid[np.newaxis, np.newaxis, ...]
        grid_tensor = torch.from_numpy(grid)

        if has_channels:
            sub_tensor = torch.from_numpy(sub_volume).unsqueeze(0)
        else:
            sub_tensor = torch.from_numpy(sub_volume).unsqueeze(0).unsqueeze(0)

        mode = 'bilinear' if interpolation != 'nearest' else 'nearest'
        sampled = F.grid_sample(
            sub_tensor,
            grid_tensor,
            mode=mode,
            padding_mode='zeros',
            align_corners=True
        )

        sampled = sampled.squeeze(0)
        if has_channels:
            sampled = sampled.squeeze(1)
        else:
            sampled = sampled.squeeze(0)

        sampled_np = sampled.numpy().astype(np.float32)

        valid_x = (coord_global[..., 0] >= 0.0) & (coord_global[..., 0] <= (depth_x - 1))
        valid_y = (coord_global[..., 1] >= 0.0) & (coord_global[..., 1] <= (depth_y - 1))
        valid_z = (coord_global[..., 2] >= 0.0) & (coord_global[..., 2] <= (depth_z - 1))
        mask_2d = (valid_x & valid_y & valid_z).astype(np.uint8)

        if has_channels:
            sampled_np *= mask_2d[np.newaxis, ...]
        else:
            sampled_np *= mask_2d

        mask_3d = None
        if return_mask:
            mask_shape = full_shape
            if len(mask_shape) == 2:
                mask_shape = (1, mask_shape[0], mask_shape[1])
            mask_3d = np.zeros(mask_shape, dtype=np.uint8)
            global_flat = coord_global.reshape(-1, 3)
            xs = np.clip(np.round(global_flat[:, 0]).astype(int), 0, mask_shape[2] - 1)
            ys = np.clip(np.round(global_flat[:, 1]).astype(int), 0, mask_shape[1] - 1)
            zs = np.clip(np.round(global_flat[:, 2]).astype(int), 0, mask_shape[0] - 1)
            mask_3d[zs, ys, xs] = 1

        return sampled_np, (mask_2d, mask_3d)

    def _use_linear_label_interp(self, target_name, plane):
        conf = self.slice_label_interpolation.get(target_name)
        if not conf:
            return False
        if isinstance(conf, dict):
            if plane in conf:
                return conf[plane]
            if '__all__' in conf:
                return conf['__all__']
        return bool(conf)

    def _initialize_slice_mask_volume_shape(self):
        first_target = list(self.target_volumes.keys())[0]
        volume_shapes = []
        for volume_info in self.target_volumes[first_target]:
            spatial_shape = self._extract_spatial_shape(volume_info['data']['data'])
            if spatial_shape is not None and len(spatial_shape) == 3:
                volume_shapes.append(tuple(int(v) for v in spatial_shape))

        if not volume_shapes:
            return

        max_shape = (
            max(shape[0] for shape in volume_shapes),
            max(shape[1] for shape in volume_shapes),
            max(shape[2] for shape in volume_shapes),
        )

        unique_shapes = {shape for shape in volume_shapes}
        if len(unique_shapes) > 1:
            print(
                "Slice plane masks will be padded/cropped to uniform shape "
                f"{max_shape} (detected volume shapes: {sorted(unique_shapes)})"
            )

        self.slice_mask_volume_shape = max_shape

    def _normalize_plane_mask_volume(self, mask_volume):
        if self.slice_plane_mask_mode != 'volume':
            return mask_volume
        target_shape = self.slice_mask_volume_shape
        if target_shape is None:
            return mask_volume
        if mask_volume is None:
            return np.zeros(target_shape, dtype=np.uint8)
        if tuple(mask_volume.shape) == tuple(target_shape) and mask_volume.dtype == np.uint8:
            return mask_volume
        mask_uint8 = mask_volume.astype(np.uint8, copy=False)
        return pad_or_crop_3d(mask_uint8, target_shape)

    def _build_axis_aligned_plane_mask_2d(self, spatial_shape, plane, slice_idx, pos0, pos1, patch_size):
        if spatial_shape is None or len(spatial_shape) != 3:
            return None

        ph, pw = patch_size
        mask = np.zeros((ph, pw), dtype=np.uint8)

        depth_z, depth_y, depth_x = spatial_shape

        if plane == 'z':
            if not (0 <= slice_idx < depth_z):
                return mask
            axis0_min, axis0_max = 0, depth_y
            axis1_min, axis1_max = 0, depth_x
        elif plane == 'y':
            if not (0 <= slice_idx < depth_y):
                return mask
            axis0_min, axis0_max = 0, depth_z
            axis1_min, axis1_max = 0, depth_x
        elif plane == 'x':
            if not (0 <= slice_idx < depth_x):
                return mask
            axis0_min, axis0_max = 0, depth_z
            axis1_min, axis1_max = 0, depth_y
        else:
            return mask

        g0_start = max(pos0, axis0_min)
        g0_end = min(pos0 + ph, axis0_max)
        g1_start = max(pos1, axis1_min)
        g1_end = min(pos1 + pw, axis1_max)

        if g0_end <= g0_start or g1_end <= g1_start:
            return mask

        local0_start = g0_start - pos0
        local1_start = g1_start - pos1

        mask[local0_start:local0_start + (g0_end - g0_start),
             local1_start:local1_start + (g1_end - g1_start)] = 1

        return mask

    def _prepare_plane_mask(self, mask_2d, mask_3d, array, plane, slice_idx, pos0, pos1, patch_size):
        if not self.slice_save_plane_masks:
            return None

        if self.slice_plane_mask_mode == 'plane':
            if mask_2d is None:
                spatial_shape = self._extract_spatial_shape(array)
                mask_2d = self._build_axis_aligned_plane_mask_2d(
                    spatial_shape=spatial_shape,
                    plane=plane,
                    slice_idx=slice_idx,
                    pos0=pos0,
                    pos1=pos1,
                    patch_size=patch_size
                )
            if mask_2d is None:
                return None
            mask_plane = mask_2d.astype(np.uint8, copy=False)
            return pad_or_crop_2d(mask_plane, patch_size)

        # Default to volume mode
        if mask_3d is None:
            mask_3d = self._build_axis_aligned_plane_mask(
                array,
                plane=plane,
                slice_idx=slice_idx,
                pos0=pos0,
                pos1=pos1,
                patch_size=patch_size
            )
        if mask_3d is None:
            return None
        return self._normalize_plane_mask_volume(mask_3d)

    def _build_axis_aligned_plane_mask(self, array, plane, slice_idx, pos0, pos1, patch_size):
        spatial_shape = self._extract_spatial_shape(array)
        if spatial_shape is None or len(spatial_shape) != 3:
            return None

        depth_z, depth_y, depth_x = spatial_shape
        mask = np.zeros(spatial_shape, dtype=np.uint8)

        if plane == 'z':
            if 0 <= slice_idx < depth_z:
                y0 = max(0, pos0)
                y1 = min(depth_y, pos0 + patch_size[0])
                x0 = max(0, pos1)
                x1 = min(depth_x, pos1 + patch_size[1])
                if y0 < y1 and x0 < x1:
                    mask[slice_idx, y0:y1, x0:x1] = 1
        elif plane == 'y':
            if 0 <= slice_idx < depth_y:
                z0 = max(0, pos0)
                z1 = min(depth_z, pos0 + patch_size[0])
                x0 = max(0, pos1)
                x1 = min(depth_x, pos1 + patch_size[1])
                if z0 < z1 and x0 < x1:
                    mask[z0:z1, slice_idx, x0:x1] = 1
        elif plane == 'x':
            if 0 <= slice_idx < depth_x:
                z0 = max(0, pos0)
                z1 = min(depth_z, pos0 + patch_size[0])
                y0 = max(0, pos1)
                y1 = min(depth_y, pos1 + patch_size[1])
                if z0 < z1 and y0 < y1:
                    mask[z0:z1, y0:y1, slice_idx] = 1

        return mask

    def _extract_slice_patch(self, patch_info):
        vol_idx = patch_info["volume_index"]
        plane = patch_info.get("plane", self.slice_primary_plane or 'z')
        slice_idx = int(patch_info.get("slice_index", 0))
        pos0, pos1 = [int(v) for v in patch_info.get("position", [0, 0])]
        patch_size = tuple(int(v) for v in patch_info.get("patch_size", self.slice_plane_patch_sizes.get(plane, self.patch_size)))

        first_target = list(self.target_volumes.keys())[0]
        img_arr = self.target_volumes[first_target][vol_idx]['data']['data']

        yaw_angle = self._sample_yaw_angle(plane)
        tilt_angles = self._sample_tilt_angles(plane)

        orientation = None
        image_mask_2d = None
        image_mask_3d = None
        if abs(yaw_angle) > 1e-6 or any(abs(v) > 1e-6 for v in tilt_angles.values()):
            orientation = self._compute_plane_orientation(
                plane=plane,
                slice_idx=slice_idx,
                pos0=pos0,
                pos1=pos1,
                patch_size=patch_size,
                yaw_angle=yaw_angle,
                tilt_angles=tilt_angles
            )
            img_patch, (image_mask_2d, image_mask_3d) = self._sample_rotated_plane(
                img_arr,
                plane=plane,
                slice_idx=slice_idx,
                pos0=pos0,
                pos1=pos1,
                patch_size=patch_size,
                orientation=orientation,
                interpolation='linear',
                return_mask=True
            )
            if img_patch is None:
                orientation = None
                image_mask_2d = None
                image_mask_3d = None
        else:
            img_patch = None

        if img_patch is None:
            img_patch = self._slice_array_patch(img_arr, plane, slice_idx, pos0, pos1, patch_size)
            if image_mask_2d is None:
                if self.slice_save_plane_masks and self.slice_plane_mask_mode == 'plane':
                    spatial_shape = self._extract_spatial_shape(img_arr)
                    image_mask_2d = self._build_axis_aligned_plane_mask_2d(
                        spatial_shape=spatial_shape,
                        plane=plane,
                        slice_idx=slice_idx,
                        pos0=pos0,
                        pos1=pos1,
                        patch_size=patch_size
                    )
                if image_mask_2d is None:
                    image_mask_2d = np.ones(patch_size, dtype=np.uint8)
            if image_mask_3d is None and self.slice_save_plane_masks and self.slice_plane_mask_mode == 'volume':
                image_mask_3d = self._build_axis_aligned_plane_mask(
                    img_arr,
                    plane=plane,
                    slice_idx=slice_idx,
                    pos0=pos0,
                    pos1=pos1,
                    patch_size=patch_size
                )
        if img_patch is None:
            img_patch = np.zeros(patch_size, dtype=np.float32)
            if image_mask_2d is None:
                image_mask_2d = np.zeros(patch_size, dtype=np.uint8)
            if image_mask_3d is None and self.slice_save_plane_masks and self.slice_plane_mask_mode == 'volume':
                spatial_shape = self._extract_spatial_shape(img_arr)
                if spatial_shape is not None:
                    image_mask_3d = np.zeros(spatial_shape, dtype=np.uint8)

        mask_for_norm = None
        if image_mask_2d is not None:
            mask_for_norm = image_mask_2d.astype(np.float32, copy=False)
            if img_patch.ndim == 2:
                img_patch = img_patch * mask_for_norm
            else:
                img_patch = img_patch * mask_for_norm[np.newaxis, ...]

        if self.normalizer is not None:
            img_patch = self.normalizer.run(img_patch, mask=mask_for_norm)
        else:
            img_patch = img_patch.astype(np.float32)

        if mask_for_norm is not None:
            if img_patch.ndim == 2:
                img_patch = img_patch * mask_for_norm
            else:
                img_patch = img_patch * mask_for_norm[np.newaxis, ...]

        if img_patch.ndim == 2:
            img_patch = img_patch[np.newaxis, ...]

        img_patch = np.ascontiguousarray(img_patch, dtype=np.float32)

        label_patches = {}
        is_unlabeled = True

        for t_name, volumes_list in self.target_volumes.items():
            label_arr = volumes_list[vol_idx]['data'].get('label')
            label_patch = None
            label_mask_2d = None
            label_mask_3d = None
            use_linear_label_interp = self._use_linear_label_interp(t_name, plane)
            if label_arr is not None:
                if orientation is not None:
                    rotated_label, (label_mask_2d, label_mask_3d) = self._sample_rotated_plane(
                        label_arr,
                        plane=plane,
                        slice_idx=slice_idx,
                        pos0=pos0,
                        pos1=pos1,
                        patch_size=patch_size,
                        orientation=orientation,
                        interpolation='linear' if use_linear_label_interp else 'nearest',
                        return_mask=True
                    )
                    if rotated_label is not None:
                        label_patch = rotated_label
                        if self.slice_save_plane_masks and label_mask_3d is not None:
                            if image_mask_3d is None:
                                image_mask_3d = label_mask_3d
                            else:
                                image_mask_3d = np.maximum(image_mask_3d, label_mask_3d)
                if label_patch is None:
                    label_patch = self._slice_array_patch(label_arr, plane, slice_idx, pos0, pos1, patch_size)
            else:
                label_patch = None

            if label_patch is None:
                label_patch = np.zeros(patch_size, dtype=np.float32)
                if label_mask_2d is not None:
                    label_patch *= label_mask_2d.astype(label_patch.dtype, copy=False)
                label_patch = label_patch[np.newaxis, ...]
            else:
                if use_linear_label_interp:
                    label_patch = (label_patch > 0.5).astype(np.float32)
                if label_patch.ndim == 2:
                    if np.any(label_patch > 0):
                        is_unlabeled = False
                    if label_mask_2d is not None:
                        label_patch *= label_mask_2d.astype(label_patch.dtype, copy=False)
                    label_patch = label_patch[np.newaxis, ...]
                else:
                    if np.any(label_patch > 0):
                        is_unlabeled = False
                    if label_mask_2d is not None:
                        label_patch *= label_mask_2d.astype(label_patch.dtype, copy=False)[np.newaxis, ...]
                label_patch = np.ascontiguousarray(label_patch, dtype=np.float32)

            label_patches[t_name] = label_patch

        result = {
            'image': torch.from_numpy(img_patch),
            'is_unlabeled': is_unlabeled
        }

        for t_name, label_patch in label_patches.items():
            result[t_name] = torch.from_numpy(label_patch)

        if self.slice_save_plane_masks:
            plane_mask = self._prepare_plane_mask(
                mask_2d=image_mask_2d,
                mask_3d=image_mask_3d,
                array=img_arr,
                plane=plane,
                slice_idx=slice_idx,
                pos0=pos0,
                pos1=pos1,
                patch_size=patch_size
            )
            if plane_mask is not None:
                mask_tensor = torch.from_numpy(plane_mask.astype(np.uint8, copy=False))
                result['plane_mask'] = mask_tensor

        angles_record = {
            'yaw_rad': float(yaw_angle),
            'tilt_x_rad': float(tilt_angles.get('x', 0.0)),
            'tilt_y_rad': float(tilt_angles.get('y', 0.0)),
            'tilt_z_rad': float(tilt_angles.get('z', 0.0)),
        }

        result['patch_info'] = {
            'plane': plane,
            'slice_index': slice_idx,
            'position': patch_info.get('position'),
            'patch_size': list(patch_size),
            'angles': angles_record
        }

        # Auto-generate auxiliary targets as in the base extractor
        try:
            regression_keys = []
            for aux_name, tinfo in getattr(self.mgr, 'targets', {}).items():
                if not tinfo.get('auxiliary_task', False):
                    continue
                task_type = str(tinfo.get('task_type', '')).lower()

                source_t = tinfo.get('source_target')
                if not source_t or source_t not in label_patches:
                    continue

                src_patch = label_patches[source_t]

                if self.is_2d_dataset:
                    binary_mask = (src_patch[0] > 0).astype(np.uint8)
                else:
                    binary_mask = (src_patch[0] > 0).astype(np.uint8)

                if task_type == 'distance_transform':
                    result[aux_name] = torch.from_numpy(compute_distance_transform(binary_mask)[np.newaxis, ...].astype(np.float32))
                elif task_type == 'surface_normals':
                    normals = compute_surface_normals_from_sdt(binary_mask)
                    result[aux_name] = torch.from_numpy(normals.astype(np.float32))
                elif task_type == 'structure_tensor':
                    tensor = compute_structure_tensor(binary_mask)
                    result[aux_name] = torch.from_numpy(tensor.astype(np.float32))
                elif task_type == 'inplane_direction':
                    direction = compute_inplane_direction(binary_mask)
                    result[aux_name] = torch.from_numpy(direction.astype(np.float32))
                else:
                    regression_keys.append(aux_name)

            if regression_keys:
                for key in regression_keys:
                    if key not in result:
                        result[key] = torch.from_numpy(np.zeros_like(img_patch, dtype=np.float32))
        except Exception as exc:
            print(f"Warning: failed to compute auxiliary targets for slice patch: {exc}")

        return result

    def __len__(self):
        return len(self.valid_patches)
            
    def _extract_patch(self, patch_info):
        """
        Extract both image and label patches from the volume and return as tensors.
        
        Parameters
        ----------
        patch_info : dict
            Dictionary containing patch position and volume index information

        Returns
        -------
        dict
            Dictionary containing:
            - 'image': torch tensor of the image patch [C, H, W] or [C, D, H, W]
            - target labels: torch tensors for each target
            - 'is_unlabeled': bool indicating if patch has no valid labels
        """
        if self.slice_sampling_enabled:
            return self._extract_slice_patch(patch_info)

        vol_idx = patch_info["volume_index"]

        # Extract coordinates based on dimensionality
        if self.is_2d_dataset:
            y, x = patch_info["position"]
            dy, dx = self.patch_size[-2:]  # Last 2 dimensions
            z, dz = 0, 0
            target_shape = (dy, dx)
        else:
            z, y, x = patch_info["position"]
            if len(self.patch_size) >= 3:
                dz, dy, dx = self.patch_size[:3]
            else:
                dy, dx = self.patch_size
                dz = 1
            target_shape = (dz, dy, dx)
        
        # Get the image data
        first_target_name = list(self.target_volumes.keys())[0]
        img_arr = self.target_volumes[first_target_name][vol_idx]['data']['data']
        
        # Extract image patch
        try:
            if self.is_2d_dataset:
                img_patch = img_arr[y:y+dy, x:x+dx]
                if img_patch.size == 0:
                    raise ValueError("Empty patch")
                img_patch = pad_or_crop_2d(img_patch, (dy, dx))
            else:
                img_patch = img_arr[z:z+dz, y:y+dy, x:x+dx]
                if img_patch.size == 0:
                    raise ValueError("Empty patch")
                img_patch = pad_or_crop_3d(img_patch, (dz, dy, dx))
        except Exception as e:
            print(f"Warning: Failed to extract image patch at vol={vol_idx}, z={z}, y={y}, x={x}: {str(e)}")
            img_patch = np.zeros(target_shape, dtype=np.float32)
        
        # Apply normalization
        if self.normalizer is not None:
            img_patch = self.normalizer.run(img_patch)
        else:
            img_patch = img_patch.astype(np.float32)
        
        # Add channel dimension
        img_patch = np.ascontiguousarray(img_patch[np.newaxis, ...])
        
        # Extract label patches and check if unlabeled
        label_patches = {}
        is_unlabeled = True  # Assume unlabeled until we find valid labels

        for t_name, volumes_list in self.target_volumes.items():
            label_arr = volumes_list[vol_idx]['data'].get('label')
            
            if label_arr is None:
                # No label exists - create zero array
                label_patch = np.zeros(target_shape, dtype=np.float32)
                # Add channel dimension for consistency with labeled data
                label_patch = label_patch[np.newaxis, ...]
            else:
                # Extract label patch
                has_channels = (self.is_2d_dataset and len(label_arr.shape) == 3) or \
                              (not self.is_2d_dataset and len(label_arr.shape) == 4)

                try:
                    if self.is_2d_dataset:
                        if has_channels:
                            label_patch = label_arr[:, y:y+dy, x:x+dx]
                        else:
                            label_patch = label_arr[y:y+dy, x:x+dx]

                        # Pad/crop
                        if has_channels:
                            n_channels = label_patch.shape[0]
                            padded = [pad_or_crop_2d(label_patch[c], (dy, dx)) for c in range(n_channels)]
                            label_patch = np.stack(padded, axis=0)
                        else:
                            label_patch = pad_or_crop_2d(label_patch, (dy, dx))
                    else:
                        if has_channels:
                            label_patch = label_arr[:, z:z+dz, y:y+dy, x:x+dx]
                        else:
                            label_patch = label_arr[z:z+dz, y:y+dy, x:x+dx]

                        # Pad/crop
                        if has_channels:
                            n_channels = label_patch.shape[0]
                            padded = [pad_or_crop_3d(label_patch[c], (dz, dy, dx)) for c in range(n_channels)]
                            label_patch = np.stack(padded, axis=0)
                        else:
                            label_patch = pad_or_crop_3d(label_patch, (dz, dy, dx))

                    # Check if patch has any non-zero labels
                    if np.any(label_patch > 0):
                        is_unlabeled = False

                except Exception as e:
                    print(f"Warning: Failed to extract label patch for {t_name}: {str(e)}")
                    label_patch = np.zeros(target_shape, dtype=np.float32)
                    # Mark that we need to add channel dimension
                    has_channels = False

                # Add channel dimension if needed
                if not has_channels:
                    label_patch = label_patch[np.newaxis, ...]

            label_patch = np.ascontiguousarray(label_patch, dtype=np.float32)
            label_patches[t_name] = label_patch


        # Build result dictionary with torch tensors
        result = {
            'image': torch.from_numpy(img_patch),
            'is_unlabeled': is_unlabeled
        }

        # Convert all label patches to tensors
        for t_name, label_patch in label_patches.items():
            result[t_name] = torch.from_numpy(label_patch)

        raw_slice_index = patch_info.get('slice_index')
        if raw_slice_index is None:
            slice_index = -1
        else:
            try:
                slice_index = int(raw_slice_index)
            except (TypeError, ValueError):
                slice_index = -1

        position_meta = patch_info.get('position')
        if position_meta is None:
            position_meta = []
        elif isinstance(position_meta, (list, tuple)):
            position_meta = [int(v) for v in position_meta]
        elif hasattr(position_meta, 'tolist'):
            position_meta = [int(v) for v in position_meta.tolist()]
        else:
            try:
                position_meta = [int(position_meta)]
            except (TypeError, ValueError):
                position_meta = []

        patch_size_meta = patch_info.get('patch_size')
        if patch_size_meta is None:
            patch_size_meta = self.patch_size
        if isinstance(patch_size_meta, (list, tuple)):
            patch_size_meta = [int(v) for v in patch_size_meta]
        elif hasattr(patch_size_meta, 'tolist'):
            patch_size_meta = [int(v) for v in patch_size_meta.tolist()]
        else:
            try:
                patch_size_meta = [int(patch_size_meta)]
            except (TypeError, ValueError):
                patch_size_meta = []

        result['patch_info'] = {
            'plane': patch_info.get('plane', 'na'),
            'slice_index': slice_index,
            'position': position_meta,
            'patch_size': patch_size_meta,
            'angles': {
                'yaw_rad': 0.0,
                'tilt_x_rad': 0.0,
                'tilt_y_rad': 0.0,
                'tilt_z_rad': 0.0,
            }
        }

        # --- Auto-generate computed targets from source labels, if configured ---
        # We generate ground-truth for tasks (e.g., distance_transform, surface_normals, inplane_direction,
        # nearest_component) based on their declared source_target. This enables related regression/regularization
        # losses to run without having to precompute and store labels on disk.
        try:
            regression_keys = []
            for aux_name, tinfo in getattr(self.mgr, 'targets', {}).items():
                # Only generate for explicitly marked auxiliary tasks
                if not tinfo.get('auxiliary_task', False):
                    continue
                task_type = str(tinfo.get('task_type', '')).lower()

                source_t = tinfo.get('source_target')
                if not source_t or source_t not in label_patches:
                    # Source not available in this batch (or not configured)
                    continue

                src_patch = label_patches[source_t]  # numpy array, shape (C, ...) float32

                # Build a binary mask from the first channel of the source label
                # Foreground is > 0 in our label convention
                if self.is_2d_dataset:
                    # (C, H, W) -> (H, W)
                    binary_mask = (src_patch[0] > 0).astype(np.uint8)
                else:
                    # (C, D, H, W) -> (D, H, W)
                    binary_mask = (src_patch[0] > 0).astype(np.uint8)

                if task_type == 'distance_transform':
                    # Compute distances according to requested type: 'signed' | 'inside' | 'outside'
                    from scipy.ndimage import distance_transform_edt
                    inside = distance_transform_edt(binary_mask)
                    outside = distance_transform_edt(1 - binary_mask)
                    dist_type = str(tinfo.get('distance_type', 'signed')).lower()

                    if dist_type == 'inside':
                        distance = inside.astype(np.float32)
                    elif dist_type == 'outside':
                        distance = outside.astype(np.float32)
                    else:  # 'signed' (default)
                        distance = (outside - inside).astype(np.float32)

                    # Add channel dimension (1, ...)
                    aux_patch = distance[np.newaxis, ...]
                    aux_patch = np.ascontiguousarray(aux_patch, dtype=np.float32)
                    result[aux_name] = torch.from_numpy(aux_patch)
                    regression_keys.append(aux_name)

                elif task_type == 'surface_normals':
                    # Compute normals from SDT; returns (2,H,W) for 2D or (3,D,H,W) for 3D
                    normals, _ = compute_surface_normals_from_sdt(binary_mask, is_2d=self.is_2d_dataset, return_sdt=False)
                    aux_patch = np.ascontiguousarray(normals, dtype=np.float32)
                    result[aux_name] = torch.from_numpy(aux_patch)
                    regression_keys.append(aux_name)
                elif task_type == 'structure_tensor':
                    aux_patch = compute_structure_tensor(
                        binary_mask,
                        self.is_2d_dataset,
                        compute_from=str(tinfo.get('compute_from', 'sdt')).lower(),
                        grad_sigma=float(tinfo.get('grad_sigma', 1.0)),
                        tensor_sigma=float(tinfo.get('tensor_sigma', 1.5)),
                        ignore_index=tinfo.get('ignore_index', -100),
                    )
                    result[aux_name] = torch.from_numpy(aux_patch)
                    regression_keys.append(aux_name)
                elif task_type == 'inplane_direction':
                    aux_patch = compute_inplane_direction(
                        binary_mask,
                        self.is_2d_dataset,
                        compute_from=str(tinfo.get('compute_from', 'sdt')).lower(),
                        grad_sigma=float(tinfo.get('grad_sigma', 1.0)),
                        tensor_sigma=float(tinfo.get('tensor_sigma', 1.5)),
                        ignore_index=tinfo.get('ignore_index', -100),
                    )
                    result[aux_name] = torch.from_numpy(aux_patch)
                    regression_keys.append(aux_name)
                elif task_type == 'nearest_component':
                    from vesuvius.models.training.auxiliary_tasks import compute_nearest_component
                    aux_patch = compute_nearest_component(
                        binary_mask,
                        self.is_2d_dataset,
                        sdf_sigma=float(tinfo.get('sdf_sigma', 0.0)),
                    )
                    result[aux_name] = torch.from_numpy(aux_patch)
                    regression_keys.append(aux_name)
                else:
                    raise ValueError('Unknown auxiliary task type')

            if regression_keys:
                result['regression_keys'] = regression_keys
        except Exception as e:
            print(f"Warning: Failed to generate auxiliary targets for patch index {index}: {e}")
        
        return result

    def _create_training_transforms(self):
        no_spatial = getattr(self.mgr, 'no_spatial', False)
        only_spatial_and_intensity = getattr(self.mgr, 'only_spatial_and_intensity', False)
            
        dimension = len(self.mgr.train_patch_size)

        if dimension == 2:
            patch_h, patch_w = self.mgr.train_patch_size
            patch_d = None  # Not used for 2D
        elif dimension == 3:
            patch_d, patch_h, patch_w = self.mgr.train_patch_size
        else:
            raise ValueError(f"Invalid patch size dimension: {dimension}. Expected 2 or 3.")

        transforms = []

        if not no_spatial:
            # Configure rotation based on patch aspect ratio
            if dimension == 2:
                if max(self.mgr.train_patch_size) / min(self.mgr.train_patch_size) > 1.5:
                    rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
                else:
                    rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
                mirror_axes = (0, 1)
            else:  # 3D
                rotation_for_DA = (-np.pi, np.pi)
                mirror_axes = (0, 1, 2)

            # Add SpatialTransform for rotations, scaling, elastic deformations
            transforms.append(
                SpatialTransform(
                    self.mgr.train_patch_size,
                    patch_center_dist_from_border=0,
                    random_crop=False,
                    p_elastic_deform=0,
                    p_rotation=0.5,
                    rotation=rotation_for_DA,
                    p_scaling=0.2,
                    scaling=(0.7, 1.4),
                    p_synchronize_scaling_across_axes=1,
                    bg_style_seg_sampling=False,  # =, mode_seg='nearest'
                    elastic_deform_magnitude=(5, 25)
                )
            )

            if mirror_axes is not None and len(mirror_axes) > 0:
                transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        if dimension == 2:
            if not only_spatial_and_intensity:
                transforms.append(RandomTransform(
                    BlankRectangleTransform(
                        rectangle_size=tuple(
                            (max(1, size // 6), size // 3) for size in self.mgr.train_patch_size
                        ),
                        rectangle_value=np.mean,
                        num_rectangles=(1, 5),
                        force_square=False,
                        p_per_sample=0.4,
                        p_per_channel=0.5
                    ), apply_probability=0.5
                ))

            # # Illumination
            # transforms.append(RandomTransform(
            #     InhomogeneousSliceIlluminationTransform(
            #         num_defects=(2, 5),
            #         defect_width=(25, 50),
            #         mult_brightness_reduction_at_defect=(0.3, 1.5),
            #         base_p=(0.2, 0.4),
            #         base_red=(0.5, 0.9),
            #         p_per_sample=1.0,
            #         per_channel=True,
            #         p_per_channel=0.5
            #     ), apply_probability=0.4
            # ))

            # Noise and blur
            if not only_spatial_and_intensity:
                transforms.append(RandomTransform(
                    GaussianNoiseTransform(
                        noise_variance=(0, 0.15),
                        p_per_channel=1,
                        synchronize_channels=True
                    ), apply_probability=0.4
                ))
                transforms.append(RandomTransform(
                    GaussianBlurTransform(
                        blur_sigma=(0.5, 1.5),
                        synchronize_channels=False,
                        synchronize_axes=False,
                        p_per_channel=0.5, benchmark=False
                    ), apply_probability=0.4
                ))

            # Brightness/contrast/gamma
            transforms.append(RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.5, 1.5)),
                    synchronize_channels=False,
                    p_per_channel=1
                ), apply_probability=0.3
            ))
            transforms.append(RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.5, 1.5)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1
                ), apply_probability=0.3
            ))
            if not only_spatial_and_intensity:
                transforms.append(RandomTransform(
                    SimulateLowResolutionTransform(
                        scale=(0.25, 1),
                        synchronize_channels=False,
                        synchronize_axes=True,
                        ignore_axes=None,
                        allowed_channels=None,
                        p_per_channel=0.5
                    ), apply_probability=0.4
                ))
            transforms.append(RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ), apply_probability=0.2
            ))
            transforms.append(RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ), apply_probability=0.4
            ))
        else:
            if not no_spatial:
                # Only add transpose transform if all three dimensions are equal
                if patch_d == patch_h == patch_w:
                    transforms.append(RandomTransform(
                        TransposeAxesTransform(allowed_axes={0, 1, 2}),
                        apply_probability=0.2
                    ))

            if not only_spatial_and_intensity:
                transforms.append(RandomTransform(
                    BlankRectangleTransform(
                        rectangle_size=tuple(
                            (max(1, size // 6), size // 3) for size in self.mgr.train_patch_size
                        ),
                        rectangle_value=np.mean,
                        num_rectangles=(1, 5),
                        force_square=False,
                        p_per_sample=0.4,
                        p_per_channel=0.5
                    ), apply_probability=0.5
                ))

                transforms.append(RandomTransform(
                    SmearTransform(
                        shift=(5, 0),
                        alpha=0.2,
                        num_prev_slices=3,
                        smear_axis=3
                    ), apply_probability=0.3
                ))

            transforms.append(RandomTransform(
                InhomogeneousSliceIlluminationTransform(
                    num_defects=(2, 5),
                    defect_width=(25, 50),
                    mult_brightness_reduction_at_defect=(0.3, 1.5),
                    base_p=(0.2, 0.4),
                    base_red=(0.5, 0.9),
                    p_per_sample=1.0,
                    per_channel=True,
                    p_per_channel=0.5
                ), apply_probability=0.4
            ))

            if not only_spatial_and_intensity:
                transforms.append(RandomTransform(
                    GaussianNoiseTransform(
                        noise_variance=(0, 0.15),
                        p_per_channel=1,
                        synchronize_channels=True
                    ), apply_probability=0.4
                ))
                transforms.append(RandomTransform(
                    GaussianBlurTransform(
                        blur_sigma=(0.5, 1.5),
                        synchronize_channels=False,
                        synchronize_axes=False,
                        p_per_channel=0.5, benchmark=False
                    ), apply_probability=0.4
                ))

            transforms.append(RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.5, 1.5)),
                    synchronize_channels=False,
                    p_per_channel=1
                ), apply_probability=0.3
            ))
            transforms.append(RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.5, 1.5)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1
                ), apply_probability=0.3
            ))
            if not only_spatial_and_intensity:
                transforms.append(RandomTransform(
                    SimulateLowResolutionTransform(
                        scale=(0.25, 1),
                        synchronize_channels=False,
                        synchronize_axes=True,
                        ignore_axes=None,
                        allowed_channels=None,
                        p_per_channel=0.5
                    ), apply_probability=0.4
                ))
            transforms.append(RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ), apply_probability=0.2
            ))
            transforms.append(RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ), apply_probability=0.4
            ))

        if no_spatial:
            print("Spatial transformations disabled (no_spatial=True)")
        
        if only_spatial_and_intensity:
            print("Only spatial and intensity augmentations enabled (only_spatial_and_intensity=True)")

        if self._needs_skeleton_transform():
            from vesuvius.models.augmentation.transforms.utils.skeleton_transform import MedialSurfaceTransform
            transforms.append(MedialSurfaceTransform(do_tube=False))
            print("Added MedialSurfaceTransform to training pipeline")

        return ComposeTransforms(transforms)
    
    def _create_validation_transforms(self):
        """
        Create validation transforms.
        For validation, we only apply skeleton transform if needed (no augmentations).
        """
        if not self._needs_skeleton_transform():
            return None

        from vesuvius.models.augmentation.transforms.utils.skeleton_transform import MedialSurfaceTransform
        
        transforms = []
        transforms.append(MedialSurfaceTransform(do_tube=False))
        print("Added MedialSurfaceTransform to validation pipeline")
        
        return ComposeTransforms(transforms)
    
    def __getitem__(self, index):
        """
        Returns a dictionary with the following format:
        {
            'image': torch.Tensor,           # Shape: [C, H, W] (2D) or [C, D, H, W] (3D)
            'is_unlabeled': bool,            # True if patch has no valid labels
            'target_name_1': torch.Tensor,   # Shape: [C, H, W] (2D) or [C, D, H, W] (3D)
            'target_name_2': torch.Tensor,   # Additional targets as configured
            ...                              # (e.g., 'ink', 'normals', 'distance_transform', etc.)
        }
        
        Where:
        - C = number of channels (usually 1 for image, can vary for targets)
        - H, W = height and width of patch
        - D = depth of patch (3D only)
        - All tensors are float32
        - Labels are 0 for background, >0 for foreground
        - Unlabeled patches will have all-zero label tensors
        """

        if not self.valid_patches:
            raise IndexError("Dataset contains no valid patches. Check data paths and patch generation settings.")
        if index >= len(self.valid_patches) or index < -len(self.valid_patches):
            # Allow optional wrap-around if explicitly enabled in config
            wrap = bool(getattr(self.mgr, 'wrap_indices', False)) or \
                   bool(getattr(self.mgr, 'wrap_dataset_indices', False)) or \
                   (hasattr(self.mgr, 'dataset_config') and bool(self.mgr.dataset_config.get('wrap_indices', False)))
            if wrap:
                index = index % len(self.valid_patches)
            else:
                raise IndexError(f"Index {index} out of range for dataset of length {len(self.valid_patches)}")
        patch_info = self.valid_patches[index]
        data_dict = self._extract_patch(patch_info)

        # if we don't want to perform augmentation on gpu , we might as well do here with cpu in the dataloader workers
        metadata = data_dict.pop('patch_info', None)
        plane_mask = data_dict.pop('plane_mask', None)

        if self.transforms is not None and not (self.is_training and getattr(self.mgr, 'augment_on_device', False)):
            data_dict = self.transforms(**data_dict)

        if metadata is not None:
            data_dict['patch_info'] = metadata
        if plane_mask is not None:
            data_dict['plane_mask'] = plane_mask

        return data_dict
