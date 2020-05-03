
import random
import string


def generate_random_string(length: int = 5, character_set: str = string.ascii_lowercase) -> str:

    if len(character_set) == 0:
        raise ValueError("Character set must not be empty")

    return ''.join(
        random.choice(character_set)
        for _ in range(length)
    )
