# PyGMH

![Build status](https://github.com/FintelmannLabDevelopmentTeam/pygmh/workflows/CI/badge.svg)

The *PyGMH* package is the reference implementation of the [*GMH* standard](https://github.com/FintelmannLabDevelopmentTeam/GMH-Spec) in Python.

## Quickstart

1. Setup environment and dependencies using [python-poetry](https://python-poetry.org/):

    ```bash
    $ poetry install
    ...
    ```

2. Enter virtualenv:

    ```bash
    $ poetry shell
    Spawning shell within /home/*user*/.cache/pypoetry/virtualenvs/pygmh-zeOtsN75-py3.7
    . /home/*user*/.cache/pypoetry/virtualenvs/pygmh-zeOtsN75-py3.7/bin/activate
    (pygmh-zeOtsN75-py3.7) $
    ```

3. Run tests:

    ```bash
    (pygmh-zeOtsN75-py3.7) $ pytest
    ...
    ```

## CLI

The CLI of this package can be invoked using the `app.py` in the project root:

```bash
(pygmh-zeOtsN75-py3.7) $ python app.py
Usage: app.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  benchmark
  info
  transcode
```
