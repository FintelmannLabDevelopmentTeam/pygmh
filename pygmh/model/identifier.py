
import re


def is_valid_identifier(identifier: str) -> bool:

    assert isinstance(identifier, str), "Identifier must be a string"

    return bool(re.compile("^[a-zA-Z0-9-_ ]+$").match(identifier))
