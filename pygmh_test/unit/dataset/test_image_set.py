
from pygmh.dataset.image_set import ImageSet
from pygmh.model import Image


def test_create_from_list():

    foo = Image(identifier="foo")
    bar = Image(identifier="bar")

    image_set = ImageSet([foo, bar])

    assert len(image_set) == 2
    assert foo == image_set.get_by_identifier("foo")
    assert bar == image_set.get_by_identifier("bar")
