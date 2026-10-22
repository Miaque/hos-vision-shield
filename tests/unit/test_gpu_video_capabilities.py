from pathlib import Path

import yaml


def test_gpu_workers_expose_video_driver_without_expanding_device_selection():
    root = Path(__file__).resolve().parents[2]
    services = yaml.safe_load((root / "compose.gpu.yaml").read_text("utf-8"))["services"]
    for name in ("file-worker", "stream-worker"):
        worker = services[name]
        assert set(worker["environment"]["NVIDIA_DRIVER_CAPABILITIES"].split(",")) == {"compute", "utility", "video"}
        devices = worker["deploy"]["resources"]["reservations"]["devices"]
        assert len(devices) == 1 and devices[0]["driver"] == "nvidia"
        assert devices[0]["device_ids"] == [worker["environment"]["GPU_DEVICE_ID"]]
        assert "privileged" not in worker
