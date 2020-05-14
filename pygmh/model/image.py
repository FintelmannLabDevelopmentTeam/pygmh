
from typing import Optional, List, Set

import numpy as np

from pygmh.model.identifier import is_valid_identifier
from pygmh.model.meta_data import MetaData
from pygmh.model.misc import Color, Vector3
from pygmh.model.segment import ImageSegment
from pygmh.model.slice import ImageSlice


class Image:
    """Representation of a volumetric image.

    The order an direction of the image volume axes interpreted for the anatomical coordinate system are:
    1. inferior -towards- superior      (z-dimension)
    2. posterior -towards- anterior     (y-dimension)
    3. left -towards- right             (x-dimension) [referring to hand-side]
    (With "towards" meaning "with increasing array index")

    (see also: http://teem.sourceforge.net/nrrd/format.html#space)

    Args:
        image_data (Optional[np.ndarray]): (Volumetric) image data.
        identifier (Optional[str]): An optional identification string for the image.
        meta_data (Optional[MetaData]): Container for image-specific meta-data.
        voxel_size (Optional[Vector3]): Size of each individual voxel within the image volume in mm.
        voxel_spacing (Optional[Vector3]): Spacing between the centers of neighbouring voxels in mm.
    """

    def __init__(
        self,
        *,

        image_data: Optional[np.ndarray] = None,

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
        self._image_segments = set()

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

        assert identifier is None or is_valid_identifier(identifier), \
            "Invalid image identifier given: " + identifier

        self._identifier = identifier

    def get_meta_data(self) -> MetaData:
        """Gets the container for the meta-data attached to the image."""

        return self._meta_data

    def get_image_data(self) -> Optional[np.ndarray]:
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
            ImageSlice(self, slice_index, identifier=slice_identifier)
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
                ImageSlice(self, slice_index, identifier=slice_identifier)
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

    def has_segment(self, identifier: str) -> bool:
        """Returns the existence of a segment with the given identifier."""

        try:
            self.get_segment(identifier)

        except KeyError:
            return False

        return True

    def get_segment(self, identifier: str) -> ImageSegment:
        """Gets the segment with the given identifier."""

        for segment in self._image_segments:

            segment: ImageSegment = segment

            if segment.get_identifier() == identifier:
                return segment

        raise KeyError("Unknown segment: '{}'".format(identifier))

    def get_segments(self) -> Set[ImageSegment]:
        """Gets all attached segment."""

        return self._image_segments.copy()

    def get_ordered_segments(self) -> List[ImageSegment]:
        """Gets all attached segments, ordered by their order of addition to the image."""

        return sorted(list(self.get_segments()), key=lambda x: x._order_index)

    def get_segment_count(self) -> int:
        """Returns the number of segments attached to this image."""

        return len(self._image_segments)

    def add_segment(self, identifier: str, mask: np.ndarray, color: Optional[Color] = None) -> ImageSegment:
        """Adds a segment with the given mask under the given identifier."""

        segment = ImageSegment(self, identifier, mask=mask, color=color)

        return self.register_segment(segment)

    def register_segment(self, image_segment: ImageSegment) -> ImageSegment:

        assert image_segment._image is self  # todo: is there a cleaner way to do this without adding public getters?
        assert not self.has_segment(image_segment.get_identifier()),\
            "There is already a segment with this identifier in this image: " + image_segment.get_identifier()

        self._image_segments.add(image_segment)

        return image_segment

    def remove_segment(self, image_segment: ImageSegment) -> None:
        """Removes the given segment from the image."""

        self._image_segments.remove(image_segment)

    def _set_image_data(self, image_data: np.ndarray) -> None:
        """Protected method to set the image data to simplify sub-classing."""

        assert isinstance(image_data, np.ndarray)
        assert np.issubdtype(image_data.dtype, np.int32),\
            "Image data must be of type np.int32. Given: " + str(image_data.dtype)
        assert image_data.ndim == 3  # we might drop this assertion in the future to allow arbitrary dimensions

        self._image_data = image_data

        # prevent after-the-fact modification of image data
        self._image_data.flags.writeable = False
