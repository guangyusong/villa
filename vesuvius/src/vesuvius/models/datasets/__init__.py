# Dataset classes for different data formats
from .base_dataset import BaseDataset
from .image_dataset import ImageDataset
from .zarr_dataset import ZarrDataset

try:
    from .napari_dataset import NapariDataset
except Exception:  # pragma: no cover - optional dependency
    NapariDataset = None


__all__ = [
    'BaseDataset',
    'ImageDataset',
    'ZarrDataset'
]

if NapariDataset is not None:
    __all__.append('NapariDataset')
