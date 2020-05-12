
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from typing import Optional

import jsonschema
import logging
import os
import tarfile

import numpy as np

from pygmh.model import Image, ImageSegment, MetaData, ImageSlice
from pygmh.persistence.interface import IAdapter
from pygmh.persistence.lazy_model import IImageDataLoader, IImageSegmentDataLoader, LazyLoadedImageSegment, LazyLoadedImage
from pygmh.util.random_string import generate_random_string


IMAGE_SEGMENT_MASK_MEMBER_NAME_FORMAT = "mask-{}.npy"


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


class Adapter(IAdapter):

    def __init__(self):

        self._logger = logging.getLogger(__name__)
        self._manifest_validation_schema: Optional[str] = None

    def read(self, path: str, *, cached: bool = False, allow_system_tar: bool = True) -> Image:

        self._logger.info("Reading gmh file from: {}".format(path))

        assert os.path.isfile(path), "Trying to read gmh file from non-file: {}".format(path)

        is_compressed = self.is_compressed(path)
        identifier = self._deduce_identifier(path)

        # use cached reader
        if cached:

            data_loader = CachedFilesystemDataLoader(path)
            manifest = data_loader.get_manifest()

        # use faster system tar if available
        elif allow_system_tar and sys.platform.startswith("linux"):

            dir_path = FilesystemDataLoader.get_temporary_directory_path()

            tar_flags = "xvf"

            if is_compressed:
                tar_flags += "z"

            return_code = subprocess.call(
                ["tar", tar_flags, path, "-C", dir_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )

            assert return_code == 0, f"Failed to extract archive to temporary path '{dir_path}'."

            with open(os.path.join(dir_path, "manifest.json"), "r") as fp:
                manifest: dict = json.load(fp)

            data_loader = FilesystemDataLoader(dir_path, manifest)

        # use default, tarfile-based mechanism
        else:

            mode = "r:gz" if is_compressed else "r:"

            tar_file_handle = tarfile.open(path, mode)

            # read manifest
            manifest_member: tarfile.TarInfo = tar_file_handle.getmember("manifest.json")
            manifest: dict = json.load(
                tar_file_handle.extractfile(manifest_member)
            )

            data_loader = TarfileDataLoader(tar_file_handle, manifest)

            # note: do not close tarfile as the handle is used within the DataLoader instance

        assert isinstance(manifest, dict)
        #self._validate_manifest(manifest)

        image = self._load_image(identifier, manifest, data_loader)

        self._load_slices(image, manifest)
        self._load_segments(image, manifest, data_loader)

        return image

    def write(self, image: Image, path: str, *, compress: bool = True, allow_system_tar: bool = True) -> None:

        self._logger.info("Writing image to gmh file under: {}".format(path))

        assert not os.path.exists(path), "Path already exists"

        # use faster system tar if available
        if allow_system_tar and sys.platform.startswith("linux"):

            self._write_using_system_tar(image, path, compress)

        # this implementation is about 15x slower because of the pure-python gzip implementation
        # see: https://github.com/ParaToolsInc/taucmdr/issues/229
        else:

            mode = "w:gz" if compress else "w"

            with tarfile.open(path, mode) as tar_file_handle:

                def add_file(name: str, content) -> None:

                    if isinstance(content, str):
                        content = content.encode()

                    member = tarfile.TarInfo(name)
                    member.size = len(content)

                    tar_file_handle.addfile(member, io.BytesIO(content))

                add_file("manifest.json", self._build_manifest_document(image))
                add_file("image_data.npy", image.get_image_data().tobytes())

                for image_segment in image.get_segments():
                    add_file(
                        IMAGE_SEGMENT_MASK_MEMBER_NAME_FORMAT.format(image_segment.get_slug()),
                        image_segment.get_mask().tobytes()
                    )

    def is_compressed(self, path: str) -> bool:
        """Derives compression from gzip header."""

        with open(path, "rb") as fp:

            header = fp.read(3)

            return header == b"\x1f\x8b\x08"

    def _deduce_identifier(self, file_path: str) -> str:
        """Deduce image identifier from file base name."""

        identifier = os.path.splitext(os.path.basename(file_path))[0]
        identifier = re.sub('[^0-9a-zA-Z]+', '_', identifier)

        return identifier

    def _load_image(self, identifier: str, manifest: dict, data_loader: AbstractDataLoader) -> Image:

        # deduce image information from manifest
        voxel_size = tuple(manifest["image"]["voxel_size"]) if manifest["image"]["voxel_size"] else None
        voxel_spacing = tuple(manifest["image"]["voxel_spacing"]) if manifest["image"]["voxel_spacing"] else None
        meta_data = MetaData(manifest["meta_data"])

        return LazyLoadedImage(data_loader, identifier, meta_data, voxel_size, voxel_spacing)

    def _load_slices(self, image: Image, manifest: dict) -> None:

        for slice_info in manifest["slices"]:

            slice_index = slice_info["index"]
            slice_identifier = slice_info["identifier"]
            slice_meta_data = MetaData(slice_info["meta_data"])

            image_slice = ImageSlice(image, slice_index, identifier=slice_identifier)
            image_slice.get_meta_data().update(slice_meta_data)

            image.register_slice(
                image_slice
            )

    def _load_segments(self, image: Image, manifest: dict, data_loader: AbstractDataLoader) -> None:

        for segment_info in manifest["segments"]:

            segment_slug = segment_info["slug"]
            segment_identifier = segment_info["identifier"]
            segment_color = tuple(segment_info["color"]) if segment_info["color"] else None
            segment_meta_data = MetaData(segment_info["meta_data"])

            segment = LazyLoadedImageSegment(image, data_loader, segment_slug, segment_identifier, segment_color)

            segment.get_meta_data().update(segment_meta_data)

            image.register_segment(
                segment
            )

    def _write_using_system_tar(self, image: Image, path: str, compressed: bool) -> None:

        target_dir_path = os.path.dirname(path)

        assert os.path.isdir(target_dir_path), "Target directory does not exist: {}".format(path)

        with tempfile.TemporaryDirectory() as temporary_dir_path:

            with open(os.path.join(temporary_dir_path, "manifest.json"), "w") as fp:
                fp.write(self._build_manifest_document(image))

            image.get_image_data().tofile(os.path.join(temporary_dir_path, "image_data.npy"))

            for image_segment in image.get_segments():

                image_segment.get_mask().tofile(
                    os.path.join(
                        temporary_dir_path,
                        IMAGE_SEGMENT_MASK_MEMBER_NAME_FORMAT.format(
                            image_segment.get_slug()
                        )
                    )
                )

            temporary_archive_path = os.path.join(target_dir_path, str(uuid.uuid4())) + ".tmp"

            tar_flags = "cvf"

            if compressed:
                tar_flags += "z"

            return_code = subprocess.call(
                ["tar", tar_flags, temporary_archive_path, "-C", temporary_dir_path] + os.listdir(temporary_dir_path),
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )

            assert return_code == 0, f"Failed to write archive to temporary path '{temporary_archive_path}'."

        os.rename(temporary_archive_path, path)

    def _build_manifest_document(self, image: Image) -> str:

        return json.dumps(self._build_manifest(image))

    def _build_manifest(self, image: Image) -> dict:

        manifest = {
            "image": {
                "precision_bytes": image.get_image_data().dtype.itemsize,
                "size": image.get_image_data().shape,  # the image volume byte sequence does not contain this
                "voxel_size": image.get_voxel_size(),
                "voxel_spacing": image.get_voxel_spacing(),
            },
            "meta_data": image.get_meta_data(),
            "slices": [
                self._build_image_slice_manifest(image_slice)
                for image_slice in image.get_slices()
            ],
            "segments": [
                self._build_image_segment_manifest(image_segment)
                for image_segment in image.get_segments()
            ],
        }

        # make sure the result will actually be readable
        #self._validate_manifest(manifest)

        return manifest

    def _build_image_slice_manifest(self, image_slice: ImageSlice) -> dict:

        return {
            "index": image_slice.get_slice_index(),
            "identifier": image_slice.get_identifier(),
            "meta_data": image_slice.get_meta_data(),
        }

    def _build_image_segment_manifest(self, image_segment: ImageSegment) -> dict:

        return {
            "slug": image_segment.get_slug(),
            "identifier": image_segment.get_identifier(),
            "color": image_segment.get_color(),
            "meta_data": image_segment.get_meta_data(),
        }

    def _validate_manifest(self, manifest: dict):

        # todo: does not work as jsonschema type "array" does not match tuples (e.g. voxel_spacing)
        # see: https://github.com/Julian/jsonschema/issues/148
        jsonschema.validate(manifest, self._get_manifest_validation_schema())

    def _get_manifest_validation_schema(self) -> str:

        if self._manifest_validation_schema is None:

            with open(os.path.dirname(__file__) + "/manifest-schema.json", "r") as file_handle:
                self._manifest_validation_schema = json.load(file_handle)

        return self._manifest_validation_schema
