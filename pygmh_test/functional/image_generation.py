
import numpy as np

from pygmh.model import Image


def generate_image() -> Image:

    image_data = np.random.rand(10, 15, 20) * pow(2, 16) - pow(2, 15)
    image_data = image_data.astype(np.int32)

    image = Image(image_data=image_data, identifier="my-identifier")
    image.set_voxel_size((1.0, 2.0, 3.0))
    image.set_voxel_spacing((2.0, 3.0, 1.0))
    image.get_meta_data().update({
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
    })

    mask1 = np.random.choice([False, True], size=image.get_image_data().shape)
    segment1 = image.add_segment("segment1", mask1)
    segment1.set_color((80, 100, 200))
    segment1.get_meta_data().update({
        "foo": "bar"
    })

    mask2 = np.random.choice([False, True], size=image.get_image_data().shape)
    segment2 = image.add_segment("segment2", mask2)
    segment2.set_color((60, 23, 150))
    segment2.get_meta_data().update({
        "foo": "baz"
    })

    slice1 = image.add_slice(0, "slice1")
    slice1.get_meta_data().update({
        "something": 1
    })

    return image
