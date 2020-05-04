
import numpy as np

from pygmh.model import Image
from pygmh.persistence.gmh import Adapter
from pygmh_test.assets import asset_path
from pygmh_test.functional.assertions import assert_equal_images


def test_read():

    adapter = Adapter()

    image = adapter.read(
        asset_path("simple_gmh/my-identifier123.gmh")
    )

    assert image.get_identifier() == "my_identifier123"
    assert np.array_equal(
        image.get_image_data(),
        np.load(asset_path("simple_gmh/image_data.npy"))
    )
    assert image.get_voxel_size() == (1.0, 2.0, 3.0)
    assert image.get_voxel_spacing() == (2.0, 3.0, 1.0)
    assert image.get_meta_data() == {
        "attr1": 1,
        "attr2": "foobar",
        "attr3": False,
        "attr4": None,
        "attr5": 2.5,
        "attr6": [
            "foo",
            "bar"
        ],
        "attr7": {
            "some": "thing"
        }
    }

    segment1 = image.get_segment("segment1")
    assert np.array_equal(
        segment1.get_mask(),
        np.load(asset_path("simple_gmh/seg1.npy"))
    )
    assert segment1.get_color() == (80, 100, 200)
    assert segment1.get_meta_data() == {
        "foo": "bar"
    }

    segment2 = image.get_segment("segment2")
    assert np.array_equal(
        segment2.get_mask(),
        np.load(asset_path("simple_gmh/seg2.npy"))
    )
    assert segment2.get_color() == (60, 23, 150)
    assert segment2.get_meta_data() == {
        "foo": "baz"
    }

    slice1 = image.get_slice(0)
    assert slice1.get_identifier() == "slice1"
    assert slice1.get_meta_data() == {
        "something": 1
    }


def test_read_write_read(tmp_path):

    adapter = Adapter()

    image = adapter.read(
        asset_path("simple_gmh/my-identifier123.gmh")
    )

    temporary_path = tmp_path / f"{image.get_identifier()}.gmh"

    adapter.write(image, temporary_path)

    image2 = adapter.read(temporary_path)

    assert_equal_images(image, image2)


def test_write_and_read(tmp_path):

    data_volume = (np.random.rand(10, 15, 20) * pow(2, 16)).astype(np.int32)

    image = Image(image_data=data_volume, identifier="test")

    adapter = Adapter()
    path = tmp_path / f"{image.get_identifier()}.gmh"

    adapter.write(image, path)
    image2 = adapter.read(path)

    assert_equal_images(image, image2)
