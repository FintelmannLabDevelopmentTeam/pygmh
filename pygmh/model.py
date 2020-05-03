"""Model for a single image.

This module contains all classes that define the in-memory model of a single image.
The model contains the following information:
    * Raw image data
    * Information to translate between array- and real-world-coordinates like voxel-size and -spacing
    * Arbitrary meta-data
    * Slice-specific information
    * A set of segmentations
"""

import re
from typing import Dict, Any, Optional, Tuple, Set, List

import numpy as np

from .util.random_string import generate_random_string


# values in millimeters
Vector3 = Tuple[float, float, float]

# 3D coordinates in some system
Coordinates3 = Tuple[int, int, int]

# RGB values
Color = Tuple[int, int, int]


class MetaData(Dict[str, Any]):
    """Container for meta-data being attached to various parts of the model."""
    pass


class ImageSegmentation:
    """Representation of a segmentation on the image.

    Args:
        image (Image): Instance of the image, the segmentation belongs to.
        identifier (str): A string-identifier for the segmentation. Has to be unique within the image instance.
        mask (np.ndarray): Boolean mask, defining the segmented area within the image.
        mask_slug (str): Identifies the mask.
        color (Optional[Color]): Default color to be used for rendering.
    """

    def __init__(
        self,
        image,  # type:Image
        identifier: str,
        mask: np.ndarray,
        mask_slug: str,
        color: Optional[Color] = None,
    ):
        assert isinstance(image, Image)

        self._image = image
        self._identifier = None
        self._mask = None
        self._mask_slug = mask_slug
        self._color = None
        self._meta_data = MetaData()

        self._order_index = max([-1] + [
            seg._order_index
            for seg in image.get_segmentations()
        ]) + 1

        self.set_identifier(identifier)
        self.set_color(color)

        if mask is not None:
            self.set_mask(mask)

    def get_identifier(self) -> str:
        """Gets the identifier of the segmentation."""
        return self._identifier

    def set_identifier(self, identifier: str) -> None:
        """Sets the identifier of the segmentation."""
        assert Image.is_valid_identifier(identifier), "Invalid segmentation identifier: " + identifier
        assert not self._image.has_segmentation(identifier),\
            "There is already a segmentation attached to the image with the given identifier: " + identifier
        self._identifier = identifier

    def get_meta_data(self) -> MetaData:
        """Gets the meta-data container of the segmentation."""
        return self._meta_data

    def get_mask(self) -> np.ndarray:
        """Gets the boolean mask of the segmentation."""
        return self._mask

    def set_mask(self, mask: np.ndarray) -> None:

        assert isinstance(mask, np.ndarray), "Given argument is not an np.ndarray"
        assert np.issubdtype(mask.dtype, np.bool_), "Mask has to be boolean. Given: " + str(mask.dtype)
        assert mask.shape == self._image.get_image_data().shape,\
            "Mask is not the same shape as the image volume. Required: {}, given: {}".format(
                str(self._image.get_image_data().shape), str(mask.shape)
            )

        self._mask = mask

        # prevent after-the-fact modification of the mask
        self._mask.flags.writeable = False

    def get_mask_slug(self) -> str:
        """Gets the mask slug."""
        return self._mask_slug

    def get_color(self) -> Optional[Color]:
        """Gets the default segmentation color as RGB tuple."""
        return self._color

    def set_color(self, color: Optional[Color]) -> None:
        """Sets the default segmentation color."""
        assert color is None or (len(color) == 3 and all(0 <= component < 256 for component in color))
        self._color = color

    def get_segmented_image_data(self,
                                 inner_substitution_value: Any = None,
                                 outer_substitution_value: Any = np.nan) -> np.ndarray:
        """Gets the raw image data with substituted values for voxels inside/outside the segmentation.

        Note:
            Passing *None* as substitution value results in no substitution taking for that region.

        Args:
            inner_substitution_value (Any): Value to substitute into for all voxels within the segmentation.
            outer_substitution_value (Any): Value to substitute into for all voxels not within the segmentation.

        Returns:
            np.array: Image data with substituted voxel values.
        """

        data = np.copy(self._image.get_image_data())
        mask = self.get_mask()

        if inner_substitution_value is not None:
            data[mask] = inner_substitution_value

        if outer_substitution_value is not None:
            data[~mask] = outer_substitution_value

        return data

    def get_bounding_box_definition(self) -> Tuple[Vector3, Vector3]:
        """Gets the tuples of indices that define a diagonal vector which further defines the bounding box"""

        non_zero = np.nonzero(self.get_mask())

        return (
            (np.min(non_zero[0]), np.min(non_zero[1]), np.min(non_zero[2])),
            (np.max(non_zero[0]), np.max(non_zero[1]), np.max(non_zero[2]))
        )

    def get_mask_in_bounding_box(self) -> np.ndarray:
        """Gets the mask defined by the bounding box"""

        bb = self.get_bounding_box_definition()

        return self.get_mask()[
            bb[0][0]:bb[1][0] + 1,
            bb[0][1]:bb[1][1] + 1,
            bb[0][2]:bb[1][2] + 1
        ]

    def get_segmented_slice_indices(self) -> Set[int]:
        """Returns the set of slice indices that the segmentation mask has"""

        indices: Set[int] = set()

        for index in np.nonzero(self._mask)[0]:
            indices.add(int(index))

        return indices


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
        identifier: Optional[str] = None,
    ):
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
        assert identifier is None or Image.is_valid_identifier(identifier),\
            "Invalid slice identifier: " + identifier
        assert identifier is None or not self._image.has_slice(identifier=identifier),\
            "There is already a slice with the given identifier: " + str(identifier)
        self._identifier = identifier

    def get_meta_data(self) -> MetaData:
        """Gets the meta-data container for the slice."""
        return self._meta_data

    def get_slice_image_data(self) -> np.ndarray:
        """Gets the raw slice image data."""
        return self._image.get_image_data()[self._slice_index]


class Image:
    """Representation of a volumetric image.

    The order an direction of the image volume axes interpreted for the anatomical coordinate system are:
    1. inferior -towards- superior      (z-dimension)
    2. posterior -towards- anterior     (y-dimension)
    3. left -towards- right             (x-dimension) [referring to hand-side]
    (With "towards" meaning "with increasing array index")

    (see also: http://teem.sourceforge.net/nrrd/format.html#space)

    Args:
        image_data (np.ndarray): (Volumetric) image data.
        identifier (Optional[str]): An optional identification string for the image.
        meta_data (MetaData): Container for image-specific meta-data.
        voxel_size (Optional[Vector3]): Size of each individual voxel within the image volume in mm.
        voxel_spacing (Optional[Vector3]): Spacing between the centers of neighbouring voxels in mm.
    """

    def __init__(
        self,
        image_data: np.ndarray,
        identifier: Optional[str] = None,
        meta_data: Optional[MetaData] = None,

        voxel_size: Optional[Vector3] = None,
        voxel_spacing: Optional[Vector3] = None,
    ):
        self._image_data = None
        self._identifier = None
        self._meta_data = meta_data or MetaData()

        self._voxel_size = None
        self._voxel_spacing = None

        self._image_slices = dict()  # indexed by slice index
        self._image_segmentations = set()

        self.set_voxel_size(voxel_size)
        self.set_voxel_spacing(voxel_spacing)

        self.set_identifier(identifier)

        if image_data is not None:
            self._set_image_data(image_data)

    def get_identifier(self) -> Optional[str]:
        """Gets a potentially defined identifier."""
        return self._identifier

    def set_identifier(self, identifier: Optional[str]) -> None:
        """Sets the identifier to the given."""
        assert identifier is None or Image.is_valid_identifier(identifier), \
            "Invalid image identifier given: " + identifier
        self._identifier = identifier

    def get_meta_data(self) -> MetaData:
        """Gets the container for the meta-data attached to the image."""
        return self._meta_data

    def get_image_data(self) -> np.ndarray:
        """Gets the raw image data."""
        return self._image_data

    def get_voxel_size(self) -> Optional[Vector3]:
        """Gets the potentially defined voxel size."""
        return self._voxel_size

    def set_voxel_size(self, voxel_size: Optional[Vector3]) -> None:
        """Sets the voxel size."""
        assert voxel_size is None or all(component > 0 for component in voxel_size),\
            "Expecting voxel dimensions to be positive"
        self._voxel_size = voxel_size

    def get_voxel_spacing(self) -> Optional[Vector3]:
        """Gets the potentially defined voxel spacing in all dimensions."""
        return self._voxel_spacing

    def set_voxel_spacing(self, voxel_spacing: Optional[Vector3]) -> None:
        """Sets the voxel spacing."""
        assert voxel_spacing is None or all(component > 0 for component in voxel_spacing),\
            "Expecting voxel spacing to be positive"
        self._voxel_spacing = voxel_spacing

    def has_slice(self, index: int = None, identifier: str = None) -> bool:
        """Returns the existence of the slice with the given index or identifier."""

        if index is not None:
            return index in self._image_slices

        try:
            self.get_slice(identifier=identifier)
        except KeyError:
            return False

        return True

    def get_slice(self, index: int = None, identifier: str = None) -> ImageSlice:
        """Gets the slice with the given index or identifier."""
        assert (index is None) != (identifier is None), "Either an index or an identifier must be given."

        # resolve by index
        if index is not None:
            if index not in self._image_slices:
                raise KeyError("Unknown slice index: {}".format(index))
            return self._image_slices[index]

        # resolve by identifier
        if identifier is not None:
            for image_slice in self._image_slices:
                if image_slice.get_identifier() == identifier:
                    return image_slice
            raise KeyError("Unknown slice identifier: {}".format(identifier))

        raise Exception()

    def get_slices(self) -> Set[ImageSlice]:
        """Returns all slices, registered to the image."""
        return set(self._image_slices.values())

    def get_ordered_slices(self) -> List[ImageSlice]:
        """Returns all slices, ordered by their slice index."""
        return list(self._image_slices.values())

    def add_slice(self, slice_index: int, slice_identifier: Optional[str] = None) -> ImageSlice:
        """Adds a slice to the image."""
        return self.register_slice(
            ImageSlice(self, slice_index, slice_identifier)
        )

    def get_or_add_slice(self, slice_index: int, slice_identifier: Optional[str] = None) -> ImageSlice:

        # find already existing slice
        if self.has_slice(slice_index):

            image_slice = self.get_slice(slice_index)

            # update the identifier if given and different from current identifier
            if slice_identifier is not None and image_slice.get_identifier() != slice_identifier:
                image_slice.set_identifier(slice_identifier)

        # create new slice
        else:
            image_slice = self.register_slice(
                ImageSlice(self, slice_index, slice_identifier)
            )

        return image_slice

    def register_slice(self, image_slice: ImageSlice) -> ImageSlice:

        assert image_slice._image is self  # todo: is there a cleaner way to do this without adding public getters?
        assert not self.has_slice(index=image_slice.get_slice_index()),\
            "There is already a slice with this index registered with this image: " + str(image_slice.get_slice_index())
        assert image_slice.get_identifier() is None or not self.has_slice(identifier=image_slice.get_identifier()), \
            "There is already a slice with this identifier in this image: " + image_slice.get_identifier()

        self._image_slices[image_slice.get_slice_index()] = image_slice

        return image_slice

    def remove_slice(self, image_slice: ImageSlice) -> None:
        """Removes the given slice from the image."""
        del(self._image_slices[image_slice.get_slice_index()])

    def has_segmentation(self, identifier: str) -> bool:
        """Returns the existence of a segmentation with the given identifier."""
        try:
            self.get_segmentation(identifier)
        except KeyError:
            return False
        return True

    def get_segmentation(self, identifier: str) -> ImageSegmentation:
        """Gets the segmentation with the given identifier."""
        for segmentation in self._image_segmentations:
            segmentation: ImageSegmentation = segmentation
            if segmentation.get_identifier() == identifier:
                return segmentation
        raise KeyError("Unknown segmentation: '{}'".format(identifier))

    def get_segmentations(self) -> Set[ImageSegmentation]:
        """Gets all attached segmentations."""
        return self._image_segmentations.copy()

    def get_ordered_segmentations(self) -> List[ImageSegmentation]:
        """Gets all attached segmentations, ordered by their order of addition to the image."""
        return sorted(list(self.get_segmentations()), key=lambda x: x._order_index)

    def get_segmentation_count(self) -> int:
        """Returns the number of segmentations attached to this image."""
        return len(self._image_segmentations)

    def add_segmentation(self, identifier: str, mask: np.ndarray, color: Optional[Color] = None) -> ImageSegmentation:
        """Adds a segmentation with the given mask under the given identifier."""

        segmentation = ImageSegmentation(self, identifier, mask, self.generate_segmentation_slug(), color)

        return self.register_segmentation(segmentation)

    def register_segmentation(self, image_segmentation: ImageSegmentation) -> ImageSegmentation:

        assert image_segmentation._image is self  # todo: is there a cleaner way to do this without adding public getters?
        assert not self.has_segmentation_slug(image_segmentation.get_mask_slug())
        assert not self.has_segmentation(image_segmentation.get_identifier()),\
            "There is already a segmentation with this identifier in this image: " + image_segmentation.get_identifier()

        self._image_segmentations.add(image_segmentation)

        return image_segmentation

    def remove_segmentation(self, image_segmentation: ImageSegmentation) -> None:
        """Removes the given segmentation from the image."""
        self._image_segmentations.remove(image_segmentation)

    def generate_segmentation_slug(self) -> str:
        """Generates a random slug which identifies the segmentation mask."""
        while True:
            slug = generate_random_string()
            if not self.has_segmentation_slug(slug):
                return slug

    def has_segmentation_slug(self, slug: str) -> bool:
        """Returns existence of segmentation slug."""
        for segmentation in self.get_segmentations():
            if segmentation.get_mask_slug() == slug:
                return True

        return False

    def _set_image_data(self, image_data: np.ndarray) -> None:
        """Protected method to set the image data to simplify sub-classing."""
        assert isinstance(image_data, np.ndarray)
        assert np.issubdtype(image_data.dtype, np.int32),\
            "Image data must be of type np.int32. Given: " + str(image_data.dtype)
        assert image_data.ndim == 3  # we might drop this assertion in the future to allow arbitrary dimensions

        self._image_data = image_data

        # prevent after-the-fact modification of image data
        self._image_data.flags.writeable = False

    @staticmethod
    def is_valid_identifier(identifier: str) -> bool:
        assert isinstance(identifier, str), "Identifier must be a string"
        return bool(re.compile("^[a-zA-Z0-9-_ ]+$").match(identifier))
