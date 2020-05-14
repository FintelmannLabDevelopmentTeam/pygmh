
from typing import Optional, Set, Tuple, Any

import numpy as np

from pygmh.model.identifier import is_valid_identifier
from pygmh.model.meta_data import MetaData
from pygmh.model.misc import Color, Coordinates3


class ImageSegment:
    """Defines a segment of the image.

    Args:
        image (Image): Instance of the image, the segment belongs to.
        identifier (str): A string-identifier for the segment. Has to be unique within the image instance.
        mask (Optional[np.ndarray]): Boolean mask, defining the segmented area within the image.
        slug (str): Identifies the mask.
        color (Optional[Color]): Default color to be used for rendering.
    """

    def __init__(
        self,
        image,  # type: Image
        identifier: str,
        slug: str,
        *,
        mask: Optional[np.ndarray] = None,
        color: Optional[Color] = None,
    ):
        from pygmh.model import Image

        assert isinstance(image, Image)

        self._image = image
        self._identifier = None
        self._mask = None
        self._slug = slug
        self._color = None
        self._meta_data = MetaData()

        self._order_index = max([-1] + [
            seg._order_index
            for seg in image.get_segments()
        ]) + 1

        self.set_identifier(identifier)
        self.set_color(color)

        if mask is not None:
            self.set_mask(mask)

    def get_identifier(self) -> str:
        """Gets the identifier of the segment."""

        return self._identifier

    def set_identifier(self, identifier: str) -> None:
        """Sets the identifier of the segment."""

        assert is_valid_identifier(identifier), "Invalid segment identifier: " + identifier
        assert not self._image.has_segment(identifier),\
            "There is already a segment attached to the image with the given identifier: " + identifier

        self._identifier = identifier

    def get_meta_data(self) -> MetaData:
        """Gets the meta-data container of the segment."""

        return self._meta_data

    def get_mask(self) -> Optional[np.ndarray]:
        """Gets the boolean mask of the segment."""

        return self._mask

    def set_mask(self, mask: Optional[np.ndarray]) -> None:

        if mask is not None:

            assert isinstance(mask, np.ndarray), "Given argument is not an np.ndarray"
            assert np.issubdtype(mask.dtype, np.bool_), "Mask has to be boolean. Given: " + str(mask.dtype)
            assert mask.shape == self._image.get_image_data().shape,\
                "Mask is not the same shape as the image volume. Required: {}, given: {}".format(
                    str(self._image.get_image_data().shape), str(mask.shape)
                )

        self._mask = mask

        if self._mask is not None:

            # prevent after-the-fact modification of the mask
            self._mask.flags.writeable = False

    def get_slug(self) -> str:
        """Gets the mask slug."""

        return self._slug

    def get_color(self) -> Optional[Color]:
        """Gets the default segment color as RGB tuple."""

        return self._color

    def set_color(self, color: Optional[Color]) -> None:
        """Sets the default segment color."""

        assert color is None or (len(color) == 3 and all(0 <= component < 256 for component in color))

        self._color = color

    def get_segmented_image_data(self,
                                 inner_substitution_value: Any = None,
                                 outer_substitution_value: Any = np.nan) -> np.ndarray:
        """Gets the raw image data with substituted values for voxels inside/outside the segment.

        Note:
            Passing *None* as substitution value results in no substitution taking for that region.

        Args:
            inner_substitution_value (Any): Value to substitute into for all voxels within the segment.
            outer_substitution_value (Any): Value to substitute into for all voxels not within the segment.

        Returns:
            np.array: Image data with substituted voxel values.
        """

        if self._image.get_image_data() is None:
            raise Exception()

        if self.get_mask() is None:
            raise Exception()

        data = np.copy(self._image.get_image_data())
        mask = self.get_mask()

        if inner_substitution_value is not None:
            data[mask] = inner_substitution_value

        if outer_substitution_value is not None:
            data[~mask] = outer_substitution_value

        return data

    def get_bounding_box(self) -> Tuple[Coordinates3, Coordinates3]:
        """Gets the tuples of indices that define a diagonal vector which further defines the bounding box"""

        if self.get_mask() is None:
            raise Exception()

        non_zero = np.nonzero(self.get_mask())

        return (
            (np.min(non_zero[0]), np.min(non_zero[1]), np.min(non_zero[2])),
            (np.max(non_zero[0]), np.max(non_zero[1]), np.max(non_zero[2]))
        )

    def get_mask_in_bounding_box(self) -> np.ndarray:
        """Gets the mask defined by the bounding box"""

        bb = self.get_bounding_box()

        return self.get_mask()[
            bb[0][0]:bb[1][0] + 1,
            bb[0][1]:bb[1][1] + 1,
            bb[0][2]:bb[1][2] + 1
        ]

    def get_segmented_slice_indices(self) -> Set[int]:
        """Returns the set of slice indices that the segment mask has."""

        if self.get_mask() is None:
            raise Exception()

        indices: Set[int] = set()

        for index in np.nonzero(self.get_mask())[0]:
            indices.add(int(index))

        return indices
