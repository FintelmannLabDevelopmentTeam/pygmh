
import tarfile
from typing import Optional, Tuple

import numpy as np

from pygmh.model import Coordinates3
from pygmh.persistence.gmh.constants import IMAGE_SEGMENT_MASK_MEMBER_NAME_FORMAT
from pygmh.persistence.lazy_model import IImageDataLoader, IImageSegmentDataLoader


class AbstractDataLoader(IImageDataLoader, IImageSegmentDataLoader):

    def __init__(self, manifest: dict):

        self._manifest = manifest

    def get_manifest(self) -> dict:

        return self._manifest

    def load_image_data(self) -> np.ndarray:

        raise NotImplementedError()

    def load_segment_mask(self, slug: str, bounding_box: Optional[Tuple[Coordinates3, Coordinates3]]) -> np.ndarray:

        raise NotImplementedError()

    def _get_image_data_dtype(self):

        return self._get_numpy_dtype_by_precision(self._manifest["image"]["precision_bytes"])

    def _get_numpy_dtype_by_precision(self, precision: int):

        if precision == 4:

            return np.int32

        else:

            raise Exception("Unknown precision: {}".format(precision))

    def _get_full_segment_mask(self, mask: np.ndarray, bounding_box: Optional[Tuple[Coordinates3, Coordinates3]]) -> np.ndarray:

        if bounding_box:

            mask = mask.reshape([
                (bounding_box[1][0] - bounding_box[0][0]) + 1,
                (bounding_box[1][1] - bounding_box[0][1]) + 1,
                (bounding_box[1][2] - bounding_box[0][2]) + 1,
            ])

            full_mask = np.zeros(self._manifest["image"]["size"], dtype=np.bool)
            full_mask[
                bounding_box[0][0]:bounding_box[1][0] + 1,
                bounding_box[0][1]:bounding_box[1][1] + 1,
                bounding_box[0][2]:bounding_box[1][2] + 1,
            ] = mask

            mask = full_mask

        else:

            mask = mask.reshape(self._manifest["image"]["size"])

        return mask


class TarfileDataLoader(AbstractDataLoader):

    def __init__(self, tar_file_handle: tarfile.TarFile, manifest: dict):

        super().__init__(manifest)

        self._tar_file_handle = tar_file_handle

    def load_image_data(self) -> np.ndarray:

        member: tarfile.TarInfo = self._tar_file_handle.getmember("image_data.npy")

        array: np.ndarray = np.frombuffer(
            self._tar_file_handle.extractfile(member).read(),
            dtype=self._get_image_data_dtype()
        )
        array = array.reshape(self._manifest["image"]["size"])

        return array

    def load_segment_mask(self, slug: str, bounding_box: Optional[Tuple[Coordinates3, Coordinates3]]) -> np.ndarray:

        member = self._tar_file_handle.getmember(
            IMAGE_SEGMENT_MASK_MEMBER_NAME_FORMAT.format(slug)
        )

        mask: np.ndarray = np.frombuffer(self._tar_file_handle.extractfile(member).read(), dtype=np.bool)

        return self._get_full_segment_mask(mask, bounding_box)
