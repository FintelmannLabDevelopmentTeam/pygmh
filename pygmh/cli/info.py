
import os

import click

from pygmh.cli.util import format_byte_count
from pygmh.persistence.gmh.adapter import Adapter


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--cached/--not-cached", default=False)
@click.option("--allow-system-tar/--disallow-system-tar", default=True)
def main(path: str, cached: bool, allow_system_tar: bool):

    adapter = Adapter()

    image = adapter.read(path, cached=cached, allow_system_tar=allow_system_tar)

    click.echo(f"Path: {path}")
    click.echo(f"Size: {format_byte_count(os.stat(path).st_size)}")
    click.echo(f"Compressed: {'Yes' if adapter.is_compressed(path) else 'No'}")
    click.echo()
    click.echo(f"Identifier: {image.get_identifier()}")
    click.echo(f"Dimensions: {image.get_image_data().shape}")
    click.echo()
    click.echo(f"Segments: {', '.join(map(lambda segment: segment.get_identifier(), image.get_segments())) if image.get_segments() else '-'}")


if __name__ == "__main__":
    main()
