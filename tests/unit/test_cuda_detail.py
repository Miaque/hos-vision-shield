import numpy as np
import pytest


def test_plate_graph_reuses_bound_addresses_and_returns_independent_outputs(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch
    from vision_shield.detection.cuda_detail import Detector

    allocations, bindings, pointers = [], [], []
    empty = torch.empty

    def allocate(*shape, **kwargs):
        assert kwargs.pop('device') == torch.device('cuda', 2)
        tensor = empty(*shape, **kwargs)
        allocations.append(tensor)
        return tensor

    class Binding:
        def bind_input(self, name, device, index, dtype, shape, pointer):
            self.input = pointer

        def bind_output(self, name, device, index, dtype, shape, pointer):
            self.output = pointer

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name='input')]

        def get_outputs(self):
            return [SimpleNamespace(name='output')]

        def io_binding(self):
            binding = Binding(); bindings.append(binding)
            return binding

        def run_with_iobinding(self, binding):
            pointers.append((binding.input, binding.output))
            source = next(t for t in allocations if t.data_ptr() == binding.input)
            target = next(t for t in allocations if t.data_ptr() == binding.output)
            target.fill_(source.flatten()[0].item())

    detector = Detector.__new__(Detector)
    detector.device = torch.device('cuda', 2)
    detector.plate_stream = SimpleNamespace(synchronize=lambda: None)
    detector.plate_bindings = {}
    detector._plate_session = lambda batch: Session()
    monkeypatch.setattr(torch, 'empty', allocate)
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    monkeypatch.setattr(torch.Tensor, 'cpu', lambda tensor: tensor.clone())
    first = detector._plate_forward(torch.ones((1, 3, 320, 320)))
    second = detector._plate_forward(torch.full((1, 3, 320, 320), 2.))
    third = detector._plate_forward(torch.full((2, 3, 320, 320), 3.))
    assert len(bindings) == 2 and len(allocations) == 4
    assert pointers[0] == pointers[1] and pointers[2] != pointers[1]
    assert first.shape == second.shape == (1, 6300, 15) and third.shape == (2, 6300, 15)
    assert (first == 1).all() and (second == 2).all() and (third == 3).all()


def test_numpy_plate_confirmation_explicitly_disables_graph_capture():
    from types import SimpleNamespace
    from vision_shield.detection.cuda_detail import Detector

    expected = np.zeros((1, 6300, 15), np.float32)
    calls = []
    detector = Detector.__new__(Detector)
    detector.plate_no_graph = object()
    detector.plate = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name='input')],
                                    run=lambda *a, **kw: calls.append((a, kw)) or [expected])
    assert detector._plate_forward(np.zeros((1, 3, 320, 320), np.float32)) is expected
    assert calls[0][1]['run_options'] is detector.plate_no_graph


def test_plate_graph_failure_propagates_without_returning_cached_output(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch
    from vision_shield.detection.cuda_detail import Detector

    def fail(binding):
        raise RuntimeError('graph failed')

    detector = Detector.__new__(Detector)
    detector.plate_stream = SimpleNamespace(synchronize=lambda: None)
    detector._plate_session = lambda batch: SimpleNamespace(run_with_iobinding=fail)
    detector.plate_bindings = {1: (object(), torch.empty((1, 3, 320, 320)), torch.ones((1, 6300, 15)))}
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    with pytest.raises(RuntimeError, match='graph failed'):
        detector._plate_forward(torch.zeros((1, 3, 320, 320)))


@pytest.mark.parametrize('shape,dtype', [((3, 3, 320, 320), 'float32'),
    ((1, 3, 64, 64), 'float32'), ((1, 3, 320, 320), 'float16')])
def test_plate_graph_rejects_changed_shape_or_precision_before_inference(shape, dtype):
    import torch
    from vision_shield.detection.cuda_detail import Detector

    with pytest.raises(ValueError, match='Unsupported plate graph input'):
        Detector.__new__(Detector)._plate_forward(torch.zeros(shape, dtype=getattr(torch, dtype)))


def test_yunet_decode_keeps_stride_coordinates_and_score():
    from vision_shield.detection.cuda_detail import decode_faces

    outputs = {}
    for stride in (8, 16, 32):
        count = (32 // stride) ** 2
        for name, size in (("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10)):
            outputs[f"{name}_{stride}"] = np.zeros((1, count, size), np.float32)
    outputs["cls_8"][0, 5, 0] = 0.81
    outputs["obj_8"][0, 5, 0] = 1
    faces = decode_faces(outputs, 32, 32, 0.3)
    assert faces.shape == (1, 15)
    np.testing.assert_allclose(faces[0, :4], [4, 4, 8, 8])
    assert faces[0, -1] == pytest.approx(0.9)
    outputs["cls_8"][0, 5, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid face"):
        decode_faces(outputs, 32, 32, 0.3)


def test_cuda_detail_refuses_cpu_only_runtime(monkeypatch):
    import sys
    from types import SimpleNamespace
    import torch
    from vision_shield.detection.cuda_detail import Detector

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "onnx", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "onnxruntime",
                        SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"]))
    with pytest.raises(RuntimeError, match="require ONNX Runtime CUDA"):
        Detector("unused-face", "unused-plate")


@pytest.mark.parametrize("provider", ["CUDAExecutionProvider", "CPUExecutionProvider"])
def test_fixed_face_sessions_keep_dimensions_device_and_bounded_lifetime(monkeypatch, provider):
    import sys
    from types import SimpleNamespace
    import torch
    from vision_shield.detection.cuda_detail import Detector

    created = []

    class Options:
        def __init__(self):
            self.dimensions, self.entries = {}, {}

        def add_free_dimension_override_by_name(self, name, value):
            self.dimensions[name] = value

        def add_session_config_entry(self, name, value):
            self.entries[name] = value

    class Session:
        def __init__(self, model, *, sess_options, providers):
            assert model == b"same-weights"
            assert providers[0][1]["device_id"] == 2
            assert 'enable_cuda_graph' not in providers[0][1]
            assert sess_options.entries["session.disable_cpu_ep_fallback"] == "1"
            self.dimensions, self.disabled = sess_options.dimensions, False
            created.append(self)

        def disable_fallback(self):
            self.disabled = True

        def get_providers(self):
            return [provider]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(SessionOptions=Options, InferenceSession=Session))
    detector = Detector.__new__(Detector)
    detector.device, detector.face_model, detector.face_sessions = torch.device("cuda", 2), b"same-weights", {}
    if provider != "CUDAExecutionProvider":
        with pytest.raises(RuntimeError, match="CUDA provider"):
            detector._face_session(384, 1280)
        assert not detector.face_sessions
        return
    first = detector._face_session(384, 1280)
    assert detector._face_session(384, 1280) is first
    assert first.dimensions == {"height": 384, "width": 1280}
    for height in (416, 448, 480, 512):
        detector._face_session(height, 1280)
    assert len(detector.face_sessions) == 4 and (384, 1280) not in detector.face_sessions
    assert len(created) == 5 and all(session.disabled for session in created)
    hot = detector._face_session(512, 1280)
    for height in (544, 576, 608, 640):
        assert detector._face_session(512, 1280) is hot
        detector._face_session(height, 1280)
    assert detector._face_session(512, 1280) is hot
    assert len(detector.face_sessions) == 4


@pytest.mark.parametrize("count,fault,parallel,kind", [(2, False, True, None), (4, False, True, None),
    (5, False, True, None), (6, False, True, None),
    (5, True, True, None), (5, False, False, None), (5, False, True, "face"),
    (5, False, True, "plate"), (5, False, True, "other")])
def test_batched_views_keep_source_mapping_tail_and_fail_on_invalid_member(monkeypatch, count, fault, parallel, kind):
    from contextlib import nullcontext
    import torch
    from vision_shield.detection import Detection
    from vision_shield.detection.cuda_detail import Detector

    detector = Detector.__new__(Detector)
    detector.tile_side, detector.plate_stream, detector.parallel = 4, object(), parallel
    bounds = [(4 * i, 0, 4 * i + 4, 4) for i in range(count)]
    detector._views = lambda *args: iter(bounds)
    detector._view = lambda pixels: pixels[::2, ::2]
    detector._plate_input = lambda view: (torch.ones((1, 3, 320, 320)), 320, 320, 0, 0)
    batches = []

    def forward(tensor):
        batches.append(len(tensor))
        predictions = np.zeros((len(tensor), 6300, 15), np.float32)
        predictions[:, 0, :5] = [160, 160, 160, 160, .9]
        predictions[:, 0, 13] = 1
        if fault:
            predictions[-1, 0, 0] = np.nan
        return predictions

    detector._plate_forward = forward
    face = Detection("face", (0, 0, 1, 1), .9)
    plate = Detection("plate", (.5, .5, 1.5, 1.5), .9)
    detector._detect_view = lambda view, kind=None: ([face] if kind == "face" else [plate] if kind == "plate" else [face, plate])
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    pixels = torch.ones((4, count * 4, 3), dtype=torch.uint8)
    if fault:
        with pytest.raises(ValueError, match="Invalid plate"):
            detector._candidates(pixels, kind=kind)
        return
    detections = detector._candidates(pixels, kind=kind)
    assert batches == ([] if not parallel or kind == "face" else [2] * (count // 2))
    kinds = [kind] if kind in {"face", "plate"} else ["face", "plate"]
    assert [d.kind for d in detections] == kinds * count
    expected = [(left, 0, left + 2, 2) if target == "face" else (left + 1, 1, left + 3, 3)
                for left, _, _, _ in bounds for target in kinds]
    np.testing.assert_allclose([d.box for d in detections], expected)


@pytest.mark.parametrize("provider", ["CUDAExecutionProvider", "CPUExecutionProvider"])
def test_plate_sessions_keep_selected_device_fixed_batch_and_original_singleton(monkeypatch, provider):
    import sys
    from types import SimpleNamespace
    import torch
    from vision_shield.detection.cuda_detail import Detector

    created = []

    class Options:
        def __init__(self):
            self.dimensions, self.entries = {}, {}

        def add_free_dimension_override_by_name(self, name, value):
            self.dimensions[name] = value

        def add_session_config_entry(self, name, value):
            self.entries[name] = value

    class Session:
        def __init__(self, model, *, sess_options, providers):
            assert model == b"batch-weights"
            assert providers[0][1]["device_id"] == 2
            assert providers[0][1]['enable_cuda_graph'] == '1'
            assert sess_options.entries["session.disable_cpu_ep_fallback"] == "1"
            self.batch, self.disabled = sess_options.dimensions["batch"], False
            created.append(self)

        def disable_fallback(self):
            self.disabled = True

        def get_providers(self):
            return [provider]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(SessionOptions=Options, InferenceSession=Session))
    detector = Detector.__new__(Detector)
    detector.device = torch.device("cuda", 2)
    detector.plate, detector.plate_model, detector.plate_sessions = object(), b"batch-weights", {}
    assert detector._plate_session(1) is detector.plate
    if provider != "CUDAExecutionProvider":
        with pytest.raises(RuntimeError, match="Plate CUDA provider"):
            detector._plate_session(4)
        assert not detector.plate_sessions
        return
    first = detector._plate_session(4)
    for batch in (2, 3, 4):
        assert detector._plate_session(batch).batch == batch
    assert detector._plate_session(4) is first and len(detector.plate_sessions) == 3
    assert len(created) == 3 and all(session.disabled for session in created)


@pytest.mark.parametrize("fault", [None, "missing", "wrong_batch"])
def test_batch_export_changes_shared_shape_metadata_and_preserves_learned_weights(fault):
    onnx = pytest.importorskip("onnx")
    from vision_shield.detection.cuda_detail import _batch_plate_model

    shape = onnx.numpy_helper.from_array(np.array([1, -1, 1], np.int64), "shape")
    weight = onnx.numpy_helper.from_array(np.array([1.25, -2.5], np.float32), "learned")
    nodes = [onnx.helper.make_node("Reshape", ["input", "shape"], ["first"]),
             onnx.helper.make_node("Reshape", ["first", "shape"], ["output"])]
    graph = onnx.helper.make_graph(nodes, "shared-shape",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 2, 2])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 12, 1])], [shape, weight],
        value_info=[onnx.helper.make_tensor_value_info("first", onnx.TensorProto.FLOAT, [1, 12, 1])])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 12)])
    if fault == "missing":
        model.graph.node[0].input[1] = "missing-shape"
    elif fault == "wrong_batch":
        model.graph.initializer[0].CopyFrom(onnx.numpy_helper.from_array(np.array([2, -1, 1], np.int64), "shape"))
    if fault:
        with pytest.raises(ValueError, match="Unsupported plate batch export"):
            _batch_plate_model(model)
        return
    result = onnx.load_from_string(_batch_plate_model(model))
    onnx.checker.check_model(result)
    assert result.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert result.graph.output[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert not result.graph.value_info
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(result.graph.initializer[0]), [0, -1, 1])
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(result.graph.initializer[1]), [1.25, -2.5])
