
import tarfile
from typing import Tuple

import numpy as np

from pygmh.model import Coordinates3, Image, ImageSegment
from pygmh.persistence.gmh.constants import IMAGE_SEGMENT_MASK_MEMBER_NAME_FORMAT
from pygmh.persistence.lazy_model import IImageDataLoader, IImageSegmentDataLoader


class ImageDataLoader(IImageDataLoader):

    def __init__(self, tar_file_handle: tarfile.TarFile, dtype: np.dtype, size: Coordinates3):

        self._tar_file_handle = tar_file_handle
        self._dtype = dtype
        self._size = size

    def load_image_data(self, image: Image) -> np.ndarray:

        member: tarfile.TarInfo = self._tar_file_handle.getmember("image_data.npy")

        array: np.ndarray = np.frombuffer(self._tar_file_handle.extractfile(member).read(), dtype=self._dtype)
        array = array.reshape(self._size)

        return array


class ImageSegmentDataLoader(IImageSegmentDataLoader):

    def __init__(self, tar_file_handle: tarfile.TarFile, slug: str, size: Coordinates3, bounding_box: Tuple[Coordinates3, Coordinates3]):

        self._tar_file_handle = tar_file_handle
        self._slug = slug
        self._size = size
        self._bounding_box = bounding_box

    def load_segment_mask(self, image_segment: ImageSegment) -> np.ndarray:

        member = self._tar_file_handle.getmember(
            IMAGE_SEGMENT_MASK_MEMBER_NAME_FORMAT.format(self._slug)
        )

        mask: np.ndarray = np.frombuffer(self._tar_file_handle.extractfile(member).read(), dtype=np.bool)

        if self._bounding_box:

            mask = mask.reshape([
                (self._bounding_box[1][0] - self._bounding_box[0][0]) + 1,
                (self._bounding_box[1][1] - self._bounding_box[0][1]) + 1,
                (self._bounding_box[1][2] - self._bounding_box[0][2]) + 1,
            ])

            full_mask = np.zeros(self._size, dtype=np.bool)
            full_mask[
                self._bounding_box[0][0]:self._bounding_box[1][0] + 1,
                self._bounding_box[0][1]:self._bounding_box[1][1] + 1,
                self._bounding_box[0][2]:self._bounding_box[1][2] + 1,
            ] = mask

            mask = full_mask

        else:

            mask = mask.reshape(self._size)

        return mask
