"""Model for a set of images."""

from typing import Set, Iterable

from pygmh.model import Image


class ImageSet(Set[Image]):
    """Represents a set of images."""

    def __init__(self, images: Iterable = None):

        super().__init__()

        if images is not None:
            for image in images:
                self.add(image)

    def add(self, element: Image) -> None:
        """Adds the given image to the set. Ensures unique identifier."""

        if element.get_identifier() is None:
            raise ValueError()

        if self.has_image_identifier(element.get_identifier()):
            raise KeyError()

        super().add(element)

    def remove_by_identifier(self, identifier: str) -> None:
        """Removes the given image from the set of images."""

        self.remove(
            self.get_by_identifier(identifier)
        )

    def get_by_identifier(self, identifier: str) -> Image:
        """Returns the image with the given identifier."""

        for image in self:
            if image.get_identifier() == identifier:
                return image

        raise KeyError("There is no image with the given identifier in this data set: {}".format(identifier))

    def has_image_identifier(self, identifier: str) -> bool:
        """Returns the existence of the image with the given identifier."""

        try:
            self.get_by_identifier(identifier)

        except KeyError:
            return False

        return True

    def get_images_having_segment(self, identifier: str):
        """Returns those images from the set that do have a segment with the given identifier."""

        return self.__class__({
            image
            for image in self
            if image.has_segment(identifier)
        })

    def get_images_not_having_segment(self, identifier: str):
        """Returns those images from the set that do *not* have a segment with the given identifier."""

        return self.__class__({
            image
            for image in self
            if not image.has_segment(identifier)
        })
