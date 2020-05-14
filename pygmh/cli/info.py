
import os

import click

from pygmh.cli.util import format_byte_count
from pygmh.persistence.gmh.adapter import Adapter


@click.command()
@click.argument("path", type=click.Path(exists=True, readable=True))
def main(path: str):

    adapter = Adapter()

    image = adapter.read(path)

    click.echo(f"Path: {path}")
    click.echo(f"Size: {format_byte_count(os.stat(path).st_size)}")
    click.echo()
    click.echo(f"Identifier: {image.get_identifier()}")
    click.echo(f"Dimensions: {image.get_image_data().shape}")
    click.echo(f"Encoding:   {image.get_image_data().dtype}")
    click.echo()
    click.echo(f"Segments: {', '.join(map(lambda segment: segment.get_identifier(), image.get_segments())) if image.get_segments() else '-'}")


if __name__ == "__main__":
    main()
