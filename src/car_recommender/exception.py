import sys

def error_message_detail(error, error_detail):
    _, _, tb = error_detail.exc_info()
    if tb is None:
        return str(error)
    return f"{tb.tb_frame.f_code.co_filename}:{tb.tb_lineno} - {error}"

class CustomException(Exception):
    def __init__(self, error, error_detail=sys):
        super().__init__(str(error))
        self.message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.message