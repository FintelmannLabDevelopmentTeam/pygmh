
import click

from pygmh.persistence.gmh.adapter import Adapter


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--cached/--not-cached", default=False)
def main(path: str, cached: bool):

    adapter = Adapter()

    image = adapter.read(path, cached=cached, allow_system_tar=False)

    click.echo(f"Path: {path}")
    click.echo(f"Compressed: {'Yes' if adapter.is_compressed(path) else 'No'}")
    click.echo(f"Identifier: {image.get_identifier()}")
    click.echo(f"Dimensions: {image.get_image_data().shape}")


if __name__ == "__main__":
    main()
