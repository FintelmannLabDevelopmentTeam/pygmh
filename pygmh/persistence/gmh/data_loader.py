
import json
import os
import shutil
import tarfile
import tempfile

import numpy as np

from pygmh.persistence.gmh.constants import IMAGE_SEGMENT_MASK_MEMBER_NAME_FORMAT
from pygmh.persistence.lazy_model import IImageDataLoader, IImageSegmentDataLoader
from pygmh.util.random_string import generate_random_string


class AbstractDataLoader(IImageDataLoader, IImageSegmentDataLoader):

    def __init__(self, manifest: dict):

        self._manifest = manifest

    def get_manifest(self) -> dict:

        return self._manifest

    def load_image_data(self) -> np.ndarray:

        raise NotImplementedError()

    def load_segment_mask(self, slug: str) -> np.ndarray:

        raise NotImplementedError()

    def _get_image_data_dtype(self):

        return self._get_numpy_dtype_by_precision(self._manifest["image"]["precision_bytes"])

    def _get_numpy_dtype_by_precision(self, precision: int):

        if precision == 4:

            return np.int32

        else:

            raise Exception("Unknown precision: {}".format(precision))


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

    def load_segment_mask(self, slug: str) -> np.ndarray:

        member = self._tar_file_handle.getmember(
            IMAGE_SEGMENT_MASK_MEMBER_NAME_FORMAT.format(slug)
        )

        mask: np.ndarray = np.frombuffer(self._tar_file_handle.extractfile(member).read(), dtype=np.bool)
        mask = mask.reshape(self._manifest["image"]["size"])

        return mask


class FilesystemDataLoader(AbstractDataLoader):

    def __init__(self, dir_path: str, manifest: dict):

        assert os.path.isdir(dir_path)

        super().__init__(manifest)

        self._dir_path = dir_path

    def __del__(self):

        shutil.rmtree(self._dir_path)

    def load_image_data(self) -> np.ndarray:

        array: np.ndarray = np.fromfile(
            os.path.join(self._dir_path, "image_data.npy"),
            dtype=self._get_image_data_dtype()
        )
        array = array.reshape(self._manifest["image"]["size"])

        return array

    def load_segment_mask(self, slug: str) -> np.ndarray:

        mask: np.ndarray = np.fromfile(
            os.path.join(self._dir_path, IMAGE_SEGMENT_MASK_MEMBER_NAME_FORMAT.format(slug)),
            dtype=np.bool
        )
        mask = mask.reshape(self._manifest["image"]["size"])

        return mask

    @staticmethod
    def get_temporary_directory_path() -> str:

        while True:

            path = os.path.join(tempfile.gettempdir(), generate_random_string())

            if not os.path.exists(path):

                os.mkdir(path)

                return path


class CachedFilesystemDataLoader(FilesystemDataLoader):

    def __init__(self, file_path: str):

        assert os.path.isfile(file_path)

        cache_dir_path = os.path.join(
            os.path.dirname(file_path),
            ".cache-" + os.path.basename(file_path)
        )

        # deduce cache state
        write_cache = False
        if not os.path.isdir(cache_dir_path):
            write_cache = True
        else:
            mtime_indicator = os.path.getmtime(cache_dir_path) < os.path.getmtime(file_path)
            ctime_indicator = os.path.getctime(cache_dir_path) < os.path.getctime(file_path)
            if mtime_indicator or ctime_indicator:
                shutil.rmtree(cache_dir_path)
                write_cache = True

        # write cache if necessary
        if write_cache:
            shutil.unpack_archive(file_path, cache_dir_path, "gztar")

        # read manifest
        with open(os.path.join(cache_dir_path, "manifest.json"), "r") as fp:
            manifest: dict = json.load(fp)

        super().__init__(cache_dir_path, manifest)

    def __del__(self):
        # to not remove cache-directory
        pass
