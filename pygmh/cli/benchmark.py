
import os
import time

import click

from pygmh.cli.util import format_byte_count
from pygmh.persistence.gmh.adapter import Adapter


@click.command()
@click.argument("path", type=click.Path(exists=True, readable=True))
@click.argument("iterations", type=click.INT, default=10)
@click.option("--cached/--not-cached", default=False)
@click.option("--allow-system-tar/--disallow-system-tar", default=True)
def main(path: str, iterations: int, cached: bool, allow_system_tar: bool):

    adapter = Adapter()

    click.echo(f"Image: {path}")
    click.echo(f"Size: {format_byte_count(os.stat(path).st_size)}")
    click.echo(f"Compressed: {'Yes' if adapter.is_compressed(path) else 'False'}")
    click.echo(f"Iterations: {iterations}")
    click.echo(f"Use cache: {'True' if cached else 'False'}")
    click.echo(f"Allow system tar: {'True' if allow_system_tar else 'False'}")
    click.echo()
    click.echo("Running benchmark...")

    time_before = time.monotonic()

    for _ in range(iterations):

        adapter.read(path, cached=cached, allow_system_tar=allow_system_tar)

    time_after = time.monotonic()

    duration = time_after - time_before

    click.echo()
    click.echo(f"Total time: {duration:.4f}s")
    click.echo(f"Time per iteration: {(duration / iterations):.4f}s")


if __name__ == "__main__":
    main()
