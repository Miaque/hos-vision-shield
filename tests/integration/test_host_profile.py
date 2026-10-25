"""Diagnostic wrappers must preserve the real codec path, including AAC and VFR."""

from contextlib import ExitStack
import os
from pathlib import Path
from threading import Event

import av
import pytest

from vision_shield.config import Settings
from vision_shield.media.video import process_video
from tests.integration.test_video_pipeline import FixedDetector


from vision_shield.detection import host_profile as profile

def signature(path):
    with av.open(path) as reader:
        video = [(frame.pts * frame.time_base, frame.width, frame.height) for frame in reader.decode(video=0)]
    with av.open(path) as reader:
        audio = [(packet.pts * packet.time_base, packet.dts * packet.time_base, bytes(packet))
                 for packet in reader.demux(audio=0) if packet.size] if reader.streams.audio else []
    return video, audio


@pytest.mark.parametrize("name", ["silent.mp4", "aac.mp4", "vfr.mp4"])
def test_profile_preserves_encoded_bytes_clocks_audio_and_each_frame(generated_media, tmp_path, name):
    source = generated_media / name
    plain, sampled = tmp_path / "plain.mp4", tmp_path / "sampled.mp4"
    config = Settings(_env_file=None).model_dump()
    process_video(source, plain, FixedDetector(), config, Event(), lambda event: None)
    collector = profile.ProfileTimes(tmp_path / "diagnostics")
    try:
        with ExitStack() as stack:
            profile.instrument(stack, collector)
            count = process_video(source, sampled, FixedDetector(), config, Event(), lambda event: None)
        assert plain.read_bytes() == sampled.read_bytes()
        assert signature(source) == signature(sampled)
        assert collector.frames == count == len(signature(source)[0])
        stats = collector.report(count)["host_stages"]
        assert stats["decode-host"]["calls"] >= count
        assert stats["encode-host"]["calls"] == count + 1  # Includes final encoder drain.
    finally:
        collector.close()


def test_inference_failure_does_not_encode_an_unprocessed_frame(generated_media, tmp_path):
    class Broken:
        def detect(self, pixels):
            raise RuntimeError("injected inference failure")

    collector = profile.ProfileTimes(tmp_path / "diagnostics")
    try:
        with ExitStack() as stack:
            profile.instrument(stack, collector)
            with pytest.raises(RuntimeError, match="injected inference failure"):
                process_video(generated_media / "silent.mp4", tmp_path / "failed.mp4", Broken(),
                              Settings(_env_file=None).model_dump(), Event(), lambda event: None)
        assert "encode-host" not in collector.report(0)["host_stages"]
        assert collector.failures["detect-pass"] == 1
    finally:
        collector.close()


@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="requires actual CUDA models and NVDEC/NVENC")
def test_gpu_batches_keep_profiled_video_and_record_all_view_regions(generated_media, tmp_path):
    import json
    from vision_shield.detection import load_detector

    config = Settings(_env_file=None, debug=True, inference_device="cuda", detection_backend="quality_yolo").model_dump()
    plain, sampled = tmp_path / "plain.mp4", tmp_path / "sampled.mp4"
    detector = load_detector(config)
    detector.detail.tile_side = 160
    process_video(generated_media / "silent.mp4", plain, detector, config, Event(), lambda e: None)
    collector = profile.ProfileTimes(tmp_path / "profile")
    try:
        with ExitStack() as stack:
            profile.instrument(stack, collector)
            detector = load_detector(config)
            detector.detail.tile_side = 160
            for name in ("primary", "detail"):
                profile.instrument_candidates(getattr(detector, name), name, collector)
            process_video(generated_media / "silent.mp4", sampled, detector, config, Event(), lambda e: None)
        assert plain.read_bytes() == sampled.read_bytes()
        assert signature(generated_media / "silent.mp4") == signature(sampled)
    finally:
        collector.close()
    rows = [json.loads(line) for line in (tmp_path / "profile/calls.jsonl").read_text().splitlines()]
    mixed = [r for r in rows if r["stage"] == "forward-detail-plate" and r.get("batched_candidate_regions")]
    assert mixed and all(r["view"] == "mixed" and r["source_region"] is None for r in mixed)
    assert all(len(r["batched_candidate_regions"]) == 7 for r in mixed)
