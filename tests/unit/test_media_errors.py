import av


def test_pyav_argument_errors_rebuild_stream_but_unsupported_formats_do_not():
    from vision_shield.media.child import RETRYABLE_STREAM_ERRORS

    # PyAV ArgumentError also inherits ValueError, unlike most I/O errors.
    assert isinstance(av.error.ArgumentError(22, "mux interrupted"), RETRYABLE_STREAM_ERRORS)
    assert isinstance(av.error.InvalidDataError(1, "packet damaged"), RETRYABLE_STREAM_ERRORS)
    assert not isinstance(ValueError("AAC required"), RETRYABLE_STREAM_ERRORS)
    assert not isinstance(RuntimeError("model inference failed"), RETRYABLE_STREAM_ERRORS)
