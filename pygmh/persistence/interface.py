
from abc import ABC, abstractmethod

from pygmh.model import Image


class IAdapter(ABC):

    @abstractmethod
    def read(self, path: str) -> Image:
        """Non-parameterized reading of the asset with the given fs path.

        This method assumes, that a somewhat complete image-record can be read from the given path.

        Args:
            path (str): Path to read the asset from.

        Returns:
            Image: The read image.
        """
        pass

    @abstractmethod
    def write(self, image: Image, path: str, *, override_if_existing: bool = False) -> None:
        """Non-parameterized writing of the image to the given fs path.

        Writes the given image to the given path.
        Assumes, a somewhat complete image-record can be written in this format.

        Args:
            image (Image): Image instance to write.
            path (str): Path to write the image to.
            override_if_existing (bool): Override file if it exists instead of failing.

        Returns:
            None
        """
        pass
