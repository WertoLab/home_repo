import typing as tp
import base64


def check_is_base64(string: str) -> bool:
    if not string.isascii():
        return False

    is_base64 = base64.b64encode(base64.b64decode(string)) == string.encode()
    return is_base64


def decode_base64_or_none(
    string: str, encoding: tp.Literal["utf-8", "ascii"]
) -> tp.Optional[str]:
    try:
        decoded_string = base64.b64decode(string).decode(encoding)
        return decoded_string
    except Exception:
        return None
