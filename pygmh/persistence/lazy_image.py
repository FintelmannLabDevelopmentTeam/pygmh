"""Lazy-loaded model for a single image."""

from abc import abstractmethod
from typing import Optional

import numpy as np

from pygmh.model import Image, ImageSegmentation, MetaData, Vector3, Color


class IImageDataLoader:

    @abstractmethod
    def load_image_data(self) -> np.ndarray:
        pass


class IImageSegmentationDataLoader:

    @abstractmethod
    def load_segmentation_mask(self, identifier: str) -> np.ndarray:
        pass


class LazyImage(Image):

    def __init__(
        self,
        image_data_provider: IImageDataLoader,
        identifier: Optional[str] = None,
        meta_data: Optional[MetaData] = None,
        voxel_size: Optional[Vector3] = None,
        voxel_spacing: Optional[Vector3] = None,
    ):
        assert isinstance(image_data_provider, IImageDataLoader)

        super().__init__(None, identifier, meta_data, voxel_size, voxel_spacing)

        self._image_data_provider = image_data_provider

    def get_image_data(self) -> np.ndarray:
        """Override accessor to retrieve image-data if not already loaded."""
        if self._image_data is None:
            self._set_image_data(
                self._image_data_provider.load_image_data()
            )
        return self._image_data


class LazyImageSegmentation(ImageSegmentation):

    def __init__(
            self,
            image,  # type:Image
            segmentation_data_loader: IImageSegmentationDataLoader,
            mask_slug: str,
            identifier: str,
            color: Optional[Color] = None
    ):
        assert isinstance(segmentation_data_loader, IImageSegmentationDataLoader)

        super().__init__(image, identifier, None, mask_slug, color)

        self._segmentation_data_loader = segmentation_data_loader

    def get_mask(self) -> np.ndarray:
        """Override accessor to retrieve mask if not already loaded."""
        if self._mask is None:
            self.set_mask(
                self._segmentation_data_loader.load_segmentation_mask(
                    self._identifier
                )
            )
        return self._mask
