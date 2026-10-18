import numpy as np

from vision_shield.redaction import redact


def test_both_classes_and_partial_overlap_are_solid_and_clipped():
    pixels = np.full((10, 20, 3), 200, dtype=np.uint8)
    result = redact(pixels, [(-2, -1, 4.2, 5.2), (15, 5, 25, 12)])
    assert (result[:6, :5] == 0).all()
    assert (result[5:, 15:] == 0).all()
    assert (result[:, 6:14] == 200).all()
    assert (pixels == 200).all()


def test_detector_maps_resized_coordinates_back_to_original_pixels():
    import torch
    from vision_shield.detection.egoblur import Detector

    detector = Detector.__new__(Detector)
    detector.max_side, detector.box_padding = 640, 0
    seen = []

    def predict(image):
        seen.append(tuple(image.shape))
        return torch.tensor([[0.0, 0.0, 320.0, 160.0]]), None, torch.tensor([0.9]), None

    detector.models = [predict]
    boxes = detector.detect(np.zeros((800, 1600, 3), dtype=np.uint8))
    assert seen == [(3, 320, 640)]
    assert boxes == [(0.0, 0.0, 800.0, 400.0)]


def test_image_orientation_is_normalized_and_metadata_not_copied(tmp_path):
    from PIL import Image
    from vision_shield.media.images import process_image

    source, target = tmp_path / "original.jpg", tmp_path / "result.jpg"
    metadata = Image.Exif()
    metadata[274] = 6
    Image.new("RGB", (30, 20), "white").save(source, exif=metadata)

    class NoTargets:
        def detect(self, pixels):
            assert pixels.shape == (30, 20, 3)
            return []

    process_image(source, target, NoTargets(), 1000)
    with Image.open(target) as result:
        assert result.size == (20, 30) and not result.getexif()
