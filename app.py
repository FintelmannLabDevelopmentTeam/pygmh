
import os
import time
from typing import Optional

import click

from pygmh.persistence.gmh.adapter import Adapter as GmhAdapter


def format_byte_count(byte_count: int) -> str:

    for unit in ['B','KiB','MiB','GiB','TiB']:

        if byte_count < 1024.0:
            break

        byte_count /= 1024.0

    return f"{byte_count:.3f}{unit}"


@click.group()
def app():
    pass


@app.command()
@click.argument("path", type=click.Path(exists=True, readable=True))
@click.argument("iterations", type=click.INT, default=10)
def benchmark(path: str, iterations: int):

    adapter = GmhAdapter()

    click.echo(f"Image: {path}")
    click.echo(f"Size: {format_byte_count(os.stat(path).st_size)}")
    click.echo(f"Iterations: {iterations}")
    click.echo()
    click.echo("Running benchmark...")

    time_before = time.monotonic()

    for _ in range(iterations):

        adapter.read(path)

    time_after = time.monotonic()

    duration = time_after - time_before

    click.echo()
    click.echo(f"Total time: {duration:.4f}s")
    click.echo(f"Time per iteration: {(duration / iterations):.4f}s")


@app.command()
@click.argument("path", type=click.Path(exists=True, readable=True))
def info(path: str):

    adapter = GmhAdapter()

    image = adapter.read(path)

    click.echo(f"Path: {path}")
    click.echo(f"Size: {format_byte_count(os.stat(path).st_size)}")
    click.echo()
    click.echo(f"Identifier: {image.get_identifier()}")
    click.echo(f"Dimensions: {image.get_image_data().shape}")
    click.echo(f"Encoding:   {image.get_image_data().dtype}")
    click.echo()
    click.echo(f"Segments: {', '.join(map(lambda segment: segment.get_identifier(), image.get_segments())) if image.get_segments() else '-'}")


@app.command()
@click.argument("path", type=click.Path(exists=True, readable=True))
@click.argument("target_path", type=click.Path(exists=False), required=False)
def transcode(path: str, target_path: Optional[str]):

    adapter = GmhAdapter()

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
    app()
