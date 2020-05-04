
import numpy as np

from pygmh.model import Image


def assert_equal_images(image_a: Image, image_b: Image):
    
    assert np.array_equal(image_a.get_image_data(), image_b.get_image_data())
    assert image_a.get_identifier() == image_b.get_identifier()
    assert image_a.get_meta_data() == image_b.get_meta_data()
    assert image_a.get_voxel_size() == image_b.get_voxel_size()
    assert image_a.get_voxel_spacing() == image_b.get_voxel_spacing()

    assert len(image_a.get_segments()) == len(image_b.get_segments())
    for first_segment in image_a.get_segments():

        assert image_b.has_segment(first_segment.get_identifier())

        second_segment = image_b.get_segment(first_segment.get_identifier())

        assert np.array_equal(first_segment.get_mask(), second_segment.get_mask())
        assert first_segment.get_color() == second_segment.get_color()
        assert first_segment.get_meta_data() == second_segment.get_meta_data()

    assert len(image_a.get_slices()) == len(image_b.get_slices())
    for first_slice in image_a.get_slices():

        assert image_b.has_slice(first_slice.get_slice_index())

        second_slice = image_b.get_slice(first_slice.get_slice_index())

        assert first_slice.get_identifier() == second_slice.get_identifier()
        assert first_slice.get_meta_data() == second_slice.get_meta_data()
