
import pytest

from pygmh.util.random_string import generate_random_string


@pytest.mark.parametrize("length", [0, 1, 5, 10])
def test_correct_length(length: int):

    result = generate_random_string(length)

    assert len(result) == length


@pytest.mark.parametrize("character_set", [
    "a",
    "abc",
])
def test_character_set(character_set: str):

    result = generate_random_string(100, character_set)

    for character in result:

        assert character in character_set


def test_empty_character_set():

    with pytest.raises(ValueError):

        generate_random_string(1, "")
