
def format_byte_count(byte_count: int) -> str:

    for unit in ['B','KiB','MiB','GiB','TiB']:

        if byte_count < 1024.0:
            break

        byte_count /= 1024.0

    return f"{byte_count:.3f}{unit}"
