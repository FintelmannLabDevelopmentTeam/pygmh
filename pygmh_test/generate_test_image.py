
import os

import numpy as np

from pygmh.persistence.gmh import Adapter
from pygmh_test.functional.image_generation import generate_image


image = generate_image()
image.set_identifier("my-identifier123")

target_base_path = os.path.join(
    os.path.dirname(__file__),
    "assets", "simple_gmh"
)

adapter = Adapter()
adapter.write(image, os.path.join(target_base_path, f"{image.get_identifier()}.gmh"))

np.save(os.path.join(target_base_path, "image_data.npy"), image.get_image_data())
np.save(os.path.join(target_base_path, "seg1.npy"), image.get_segment("segment1").get_mask())
np.save(os.path.join(target_base_path, "seg2.npy"), image.get_segment("segment2").get_mask())
