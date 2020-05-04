"""Lazy-loaded model for a single image."""

from abc import abstractmethod
from typing import Optional

import numpy as np

from pygmh.model import Image, ImageSegment, MetaData, Vector3, Color


class IImageDataLoader:

    @abstractmethod
    def load_image_data(self) -> np.ndarray:
        pass


class IImageSegmentDataLoader:

    @abstractmethod
    def load_segment_mask(self, identifier: str) -> np.ndarray:
        pass


class LazyLoadedImage(Image):

    def __init__(
        self,
        image_data_provider: IImageDataLoader,
        identifier: Optional[str] = None,
        meta_data: Optional[MetaData] = None,
        voxel_size: Optional[Vector3] = None,
        voxel_spacing: Optional[Vector3] = None,
    ):
        assert isinstance(image_data_provider, IImageDataLoader)

        super().__init__(identifier=identifier, meta_data=meta_data, voxel_size=voxel_size, voxel_spacing=voxel_spacing)

        self._image_data_provider = image_data_provider

    def get_image_data(self) -> np.ndarray:
        """Override accessor to retrieve image-data if not already loaded."""
        if self._image_data is None:
            self._set_image_data(
                self._image_data_provider.load_image_data()
            )
        return self._image_data


class LazyLoadedImageSegment(ImageSegment):

    def __init__(
            self,
            image,  # type:Image
            segment_data_loader: IImageSegmentDataLoader,
            slug: str,
            identifier: str,
            color: Optional[Color] = None
    ):
        assert isinstance(segment_data_loader, IImageSegmentDataLoader)

        super().__init__(image, identifier, slug, color=color)

        self._segment_data_loader = segment_data_loader

    def get_mask(self) -> np.ndarray:
        """Override accessor to retrieve mask if not already loaded."""
        if self._mask is None:
            self.set_mask(
                self._segment_data_loader.load_segment_mask(
                    self._identifier
                )
            )
        return self._mask
