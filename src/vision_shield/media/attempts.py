"""Safe structured failure evidence; exception text may contain camera credentials."""

import traceback


def failure_details(error, phase):
    result = dict(reason=f"{phase}_error", phase=phase, error_type=type(error).__name__,
                  at=[f"{frame.name}:{frame.lineno}" for frame in traceback.extract_tb(error.__traceback__)][-12:])
    if isinstance(getattr(error, "errno", None), int):
        result["errno"] = error.errno
    response = getattr(error, "response", None)
    if response is not None and isinstance(getattr(response, "status_code", None), int):
        result["http_status"] = response.status_code
    return result
