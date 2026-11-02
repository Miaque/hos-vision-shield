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


def test_test_workers_use_blackwell_build_without_changing_production():
    root = Path(__file__).resolve().parents[2]
    production = yaml.safe_load((root / "compose.gpu.yaml").read_text("utf-8"))["services"]
    testing = yaml.safe_load((root / "compose.test.yaml").read_text("utf-8"))["services"]
    for name in ("file-worker", "stream-worker"):
        assert production[name]["image"] == "vision-shield:cu126"
        assert production[name]["build"]["args"]["TORCH_BACKEND"] == "cu126"
        assert testing[name]["image"] == "vision-shield-test:cu130"
        assert testing[name]["build"]["args"]["TORCH_BACKEND"] == "cu130"
        assert testing[name]["group_add"] == ["1022"]
