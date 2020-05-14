
import os
from typing import Optional

import click

from pygmh.persistence.gmh.adapter import Adapter


@click.command()
@click.argument("path", type=click.Path(exists=True, readable=True))
@click.argument("target_path", type=click.Path(exists=False), required=False)
@click.option("--compress/--not-compress", default=False)
def main(path: str, target_path: Optional[str], compress: bool):

    adapter = Adapter()

    image = adapter.read(path)

    backup_path = None
    if target_path is None:

        # backup original file
        backup_path = f"{path}.bak"
        os.rename(path, backup_path)

    # re-write image
    adapter.write(image, target_path if target_path else path, compress=compress)

    # delete backup
    if backup_path:

        os.unlink(backup_path)


if __name__ == "__main__":
    main()
