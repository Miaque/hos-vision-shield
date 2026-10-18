from types import SimpleNamespace

import pytest

from vision_shield.config import Settings


@pytest.mark.parametrize("missing", ["face.jit", "plate.jit"])
async def test_quality_worker_rejects_missing_primary_before_receiver_start(tmp_path, monkeypatch, missing):
    from vision_shield.jobs import worker

    for name in ["face.onnx", "plate.onnx", "face.jit", "plate.jit"]:
        if name != missing:
            (tmp_path / name).write_bytes(b"present")
    settings = Settings(_env_file=None, detection_backend="quality", face_model_path=tmp_path / "face.onnx",
                        plate_model_path=tmp_path / "plate.onnx", egoblur_face_model_path=tmp_path / "face.jit",
                        egoblur_plate_model_path=tmp_path / "plate.jit")
    monkeypatch.setattr(worker, "Settings", lambda: settings)
    monkeypatch.setattr(worker, "file_broker", SimpleNamespace(connection_pool=None))

    def receiver(*args, **kwargs):
        raise AssertionError("Receiver must not start with missing quality models")

    monkeypatch.setattr(worker, "Receiver", receiver)
    with pytest.raises(RuntimeError, match="model files"):
        await worker.main("files")
