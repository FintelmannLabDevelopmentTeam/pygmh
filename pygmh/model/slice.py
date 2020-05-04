
from typing import Optional

import numpy as np

from pygmh.model.identifier import is_valid_identifier
from pygmh.model.meta_data import MetaData


class ImageSlice:
    """Representation of a slice within the image volume.

    This model captures all information regarding a single slice within the image.

    Args:
        image (Image): Instance of the image, the slice belongs to.
        slice_index (int): Index of slice in the z-axis of the image-data.
        identifier (Optional[str]): An optional identifier for the slice.
    """

    def __init__(
            self,
            image,  # type:Image
            slice_index: int,
            *,
            identifier: Optional[str] = None,
    ):
        from pygmh.model import Image

        assert isinstance(image, Image)
        assert 0 <= slice_index < image.get_image_data().shape[0]

        self._image = image
        self._slice_index = slice_index
        self._identifier = None
        self._meta_data = MetaData()

        self.set_identifier(identifier)

    def get_slice_index(self) -> int:
        """Gets the index of the slice within the image volume."""

        return self._slice_index

    def get_identifier(self) -> Optional[str]:
        """Gets the identifier of the slice or *None* if no identifier has been defined."""

        return self._identifier

    def set_identifier(self, identifier: Optional[str]) -> None:
        """Sets the identifier to the given."""

        assert identifier is None or is_valid_identifier(identifier), \
            "Invalid slice identifier: " + identifier
        assert identifier is None or not self._image.has_slice(identifier=identifier), \
            "There is already a slice with the given identifier: " + str(identifier)

        self._identifier = identifier

    def get_meta_data(self) -> MetaData:
        """Gets the meta-data container for the slice."""

        return self._meta_data

    def get_slice_image_data(self) -> np.ndarray:
        """Gets the raw slice image data."""

        if self._image.get_image_data() is None:
            raise Exception()

        return self._image.get_image_data()[self._slice_index]
