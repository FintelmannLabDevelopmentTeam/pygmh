
import io
import json
import re
from typing import Optional

import jsonschema
import logging
import os
import tarfile

import numpy as np

from pygmh.model import Image, ImageSegment, MetaData, ImageSlice
from pygmh.persistence.gmh.constants import IMAGE_SEGMENT_MASK_MEMBER_NAME_FORMAT
from pygmh.persistence.gmh.data_loader import TarfileDataLoader, AbstractDataLoader
from pygmh.persistence.interface import IAdapter
from pygmh.persistence.lazy_model import LazyLoadedImageSegment, LazyLoadedImage


class Adapter(IAdapter):

    def __init__(self):

        self._logger = logging.getLogger(__name__)
        self._manifest_validation_schema: Optional[str] = None

    def read(self, path: str) -> Image:

        self._logger.info("Reading gmh file from: {}".format(path))

        assert os.path.isfile(path), "Trying to read gmh file from non-file: {}".format(path)

        identifier = self._deduce_identifier(path)

        # note: do not close tarfile as the handle is used within the DataLoader instance
        tar_file_handle = tarfile.open(path, "r")

        # read manifest
        manifest_member: tarfile.TarInfo = tar_file_handle.getmember("manifest.json")
        manifest: dict = json.load(
            tar_file_handle.extractfile(manifest_member)
        )

        self._validate_manifest(manifest)

        data_loader = TarfileDataLoader(tar_file_handle, manifest)

        image = self._load_image(identifier, manifest, data_loader)

        self._load_slices(image, manifest)
        self._load_segments(image, manifest, data_loader)

        return image

    def write(self, image: Image, path: str) -> None:

        self._logger.info("Writing image to gmh file under: {}".format(path))

        assert not os.path.exists(path), "Path already exists"

        with tarfile.open(path, "w") as tar_file_handle:

            def add_file(name: str, content) -> None:

                if isinstance(content, str):
                    content = content.encode()

                member = tarfile.TarInfo(name)
                member.size = len(content)

                tar_file_handle.addfile(member, io.BytesIO(content))

            add_file("manifest.json", self._build_manifest_document(image))
            add_file("image_data.npy", image.get_image_data().tobytes())

            for image_segment in image.get_segments():

                if image_segment.is_empty():
                    continue

                add_file(
                    IMAGE_SEGMENT_MASK_MEMBER_NAME_FORMAT.format(image_segment.get_slug()),
                    image_segment.get_mask_in_bounding_box().tobytes()
                )

    def _deduce_identifier(self, file_path: str) -> str:
        """Deduce image identifier from file base name."""

        identifier = os.path.splitext(os.path.basename(file_path))[0]
        identifier = re.sub('[^0-9a-zA-Z-_ ]', '_', identifier)

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
            segment_bounding_box = segment_info["bounding_box"] if "bounding_box" in segment_info else None # todo: make obligatory
            segment_identifier = segment_info["identifier"]
            segment_color = tuple(segment_info["color"]) if segment_info["color"] else None
            segment_meta_data = MetaData(segment_info["meta_data"])

            # lazy-load non-empty mask
            if segment_slug is not None:

                segment = LazyLoadedImageSegment(image, data_loader, segment_bounding_box, segment_slug, segment_identifier, segment_color)

                # todo: remove
                try:
                    segment.get_mask()
                except FileNotFoundError:
                    segment = LazyLoadedImageSegment(image, data_loader, segment_bounding_box, segment_identifier, segment_identifier, segment_color)

            # construct empty mask
            else:

                mask = np.zeros(image.get_image_data().shape, dtype=np.bool)

                segment = ImageSegment(image, segment_identifier, None, mask=mask, color=segment_color)

            segment.get_meta_data().update(segment_meta_data)

            image.register_segment(
                segment
            )

    def _build_manifest_document(self, image: Image) -> str:

        return json.dumps(self._build_manifest(image))

    def _build_manifest(self, image: Image) -> dict:

        manifest = {
            "image": {
                "precision_bytes": image.get_image_data().dtype.itemsize,
                "size": list(image.get_image_data().shape),  # the image volume byte sequence does not contain this
                "voxel_size": list(image.get_voxel_size()) if image.get_voxel_size() else None,
                "voxel_spacing": list(image.get_voxel_spacing()) if image.get_voxel_spacing() else None,
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
        self._validate_manifest(manifest)

        return manifest

    def _build_image_slice_manifest(self, image_slice: ImageSlice) -> dict:

        return {
            "index": image_slice.get_slice_index(),
            "identifier": image_slice.get_identifier(),
            "meta_data": image_slice.get_meta_data(),
        }

    def _build_image_segment_manifest(self, image_segment: ImageSegment) -> dict:

        segment_manifest = {
            "bounding_box": None,
            "slug": None,
            "identifier": image_segment.get_identifier(),
            "color": list(image_segment.get_color()) if image_segment.get_color() else None,
            "meta_data": image_segment.get_meta_data(),
        }

        if not image_segment.is_empty():

            bounding_box = image_segment.get_bounding_box()

            segment_manifest["bounding_box"] = [list(bounding_box[0]), list(bounding_box[1])]
            segment_manifest["slug"] = image_segment.get_slug()

        return segment_manifest

    def _validate_manifest(self, manifest: dict):

        jsonschema.validate(manifest, self._get_manifest_validation_schema())

    def _get_manifest_validation_schema(self) -> str:

        if self._manifest_validation_schema is None:

            with open(os.path.dirname(__file__) + "/manifest-schema.json", "r") as file_handle:
                self._manifest_validation_schema = json.load(file_handle)

        return self._manifest_validation_schema
