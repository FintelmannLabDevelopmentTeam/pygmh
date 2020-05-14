
import os
from typing import Optional

import click

from pygmh.persistence.gmh.adapter import Adapter


@click.command()
@click.argument("path", type=click.Path(exists=True, readable=True))
@click.argument("target_path", type=click.Path(exists=False), required=False)
def main(path: str, target_path: Optional[str]):

    adapter = Adapter()

    image = adapter.read(path)

    temporary_target_path = None
    if target_path is None:

        # write to temporary file fist
        temporary_target_path = f"{path}.tmp"

    # re-write image
    adapter.write(image, temporary_target_path if temporary_target_path else target_path)

    # swap-in file
    if temporary_target_path:

        os.unlink(path)
        os.rename(temporary_target_path, path)


if __name__ == "__main__":
    main()
