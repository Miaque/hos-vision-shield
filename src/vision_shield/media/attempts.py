"""Safe structured failure evidence; exception text may contain camera credentials."""

import re
import traceback

from av.error import FFmpegError


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
    """保留实际失败阶段和安全协议原因，不保存可能含来源凭据的异常全文。"""
    result = dict(reason=f"{phase}_error", phase=phase, error_type=type(error).__name__,
                  at=[f"{frame.name}:{frame.lineno}" for frame in traceback.extract_tb(error.__traceback__)][-12:])
    if isinstance(getattr(error, "errno", None), int):
        result["errno"] = error.errno
    response = getattr(error, "response", None)
    if response is not None and isinstance(getattr(response, "status_code", None), int):
        result["http_status"] = response.status_code
    if isinstance(error, FFmpegError):
        for key, value in (
            ("ffmpeg_error", error.strerror),
            ("ffmpeg_log", f"[{error.log[1]}] {error.log[2]}" if error.log else None),
        ):
            if value:
                # 先隐藏完整地址及独立凭据再截断，避免截断后的半个秘密逃过过滤。
                value = re.sub(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s\"'<>]+", "<已隐藏地址>", value)
                value = re.sub(r"(?im)\b(?:Authorization|Proxy-Authorization|Cookie|Set-Cookie)\s*:[^\r\n]+",
                               "<已隐藏凭据>", value)
                value = re.sub(r'''(?i)\b(token|sign|password|secret|api[_-]?key)\s*[:=]\s*(?:"[^"]*"|'[^']*'|[^\s&;,]+)''',
                               r"\1=<已隐藏凭据>", value)
                result[key] = " ".join(value.split())[:1024]
    return result
