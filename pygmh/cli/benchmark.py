
import os
import time

import click

from pygmh.cli.util import format_byte_count
from pygmh.persistence.gmh.adapter import Adapter


@click.command()
@click.argument("path", type=click.Path(exists=True, readable=True))
@click.argument("iterations", type=click.INT, default=10)
def main(path: str, iterations: int):

    adapter = Adapter()

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


if __name__ == "__main__":
    main()
