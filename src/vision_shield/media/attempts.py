"""Safe structured failure evidence; exception text may contain camera credentials."""

import traceback


def model_runtime_failure(error):
    """Recycle model/device execution faults; input/config validation remains terminal."""
    from vision_shield.detection.preflight import DeviceUnavailableError, PreflightError

    if isinstance(error, DeviceUnavailableError):
        return True
    if isinstance(error, (PreflightError, ValueError, TypeError)):
        return False
    module, name = type(error).__module__, type(error).__name__
    if module == 'torch' or module.startswith('torch.'):
        return True
    if module.startswith('onnxruntime.'):
        return name in {'Fail', 'EPFail', 'EngineError', 'RuntimeException'}
    if module == 'av.error' and name == 'ExternalError':
        return True
    if isinstance(error, RuntimeError) or (module == 'cv2' and name == 'error'):
        frame = error.__traceback__
        while frame is not None:
            origin = frame.tb_frame.f_globals.get('__name__', '')
            if origin.startswith(('vision_shield.detection.', 'torch.')) or origin == 'vision_shield.media.cuda_frames':
                return True
            frame = frame.tb_next
    return False


def failure_details(error, phase):
    result = dict(reason=f"{phase}_error", phase=phase, error_type=type(error).__name__,
                  at=[f"{frame.name}:{frame.lineno}" for frame in traceback.extract_tb(error.__traceback__)][-12:])
    if isinstance(getattr(error, "errno", None), int):
        result["errno"] = error.errno
    response = getattr(error, "response", None)
    if response is not None and isinstance(getattr(response, "status_code", None), int):
        result["http_status"] = response.status_code
    return result
