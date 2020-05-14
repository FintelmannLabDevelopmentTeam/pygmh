"""Lazy-loaded model."""

from abc import abstractmethod

import numpy as np

from pygmh.model import Image, ImageSegment


class IImageDataLoader:

    @abstractmethod
    def load_image_data(self, image: Image) -> np.ndarray:
        pass


class IImageSegmentDataLoader:

    @abstractmethod
    def load_segment_mask(self, image_segment: ImageSegment) -> np.ndarray:
        pass


class LazyLoadedImage(Image):

    def __init__(self, image_data_loader: IImageDataLoader):

        super().__init__()

        self._image_data_loader = image_data_loader

    def get_image_data(self) -> np.ndarray:
        """Override accessor to retrieve image-data if not already loaded."""

        if self._image_data is None:

            self._set_image_data(
                self._image_data_loader.load_image_data(self)
            )

        return self._image_data


class LazyLoadedImageSegment(ImageSegment):

    def __init__(self, segment_data_loader: IImageSegmentDataLoader, image: Image, identifier: str):

        assert isinstance(segment_data_loader, IImageSegmentDataLoader)

        super().__init__(image, identifier)

        self._segment_data_loader = segment_data_loader

    def get_mask(self) -> np.ndarray:
        """Override accessor to retrieve mask if not already loaded."""

        if self._mask is None:

            self.set_mask(
                self._segment_data_loader.load_segment_mask(self)
            )

        return self._mask
