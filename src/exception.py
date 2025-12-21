import sys
from types import TracebackType

def error_message_detail(
    error: Exception,
    exc_info: tuple[type, BaseException, TracebackType|None] | None = None
) -> str:
    """
    Build a helpful one-line error message including filename and lineno.
    exc_info should be something like sys.exc_info(); if omitted, sys.exc_info() is used.
    """
    exc_info = exc_info or sys.exc_info()
    _, _, tb = exc_info
    if tb is None:
        return f"Error message [{error}]"

    # Walk to the last frame to show the origin
    while tb.tb_next:
        tb = tb.tb_next
    file_name = tb.tb_frame.f_code.co_filename
    line_no = tb.tb_lineno
    return f"Error occurred in python script name [{file_name}] line number [{line_no}] error message [{error}]"

class CustomException(Exception):
    """
    Lightweight wrapper that stores a decorated error message.
    Prefer: `raise CustomException(msg) from original_exc` to preserve traceback.
    Or: `raise CustomException(e, sys.exc_info())` if you need the built message.
    """
    def __init__(
        self,
        error_message,
        exc_info: tuple[type, BaseException, TracebackType|None] | None = None
    ):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, exc_info)

    def __str__(self) -> str:
        return self.error_message