"""Replay unchanged Gen1 backbone kernels; dynamic proposals and NMS stay eager."""
from weakref import WeakValueDictionary

import torch


_owners = WeakValueDictionary()
_library = torch.library.Library('vision_shield_plate', 'DEF')
_library.define('backbone(Tensor pixels, int owner) -> (Tensor, Tensor, Tensor, Tensor, Tensor)')
_library.impl('backbone', lambda pixels, owner: _owners[owner].replay(pixels), 'CUDA')


def replace_backbone(graph, owner):
    calls = [node for node in graph.nodes() if node.kind() == 'prim::CallMethod'
             and node.inputsAt(0).node().kind() == 'prim::GetAttr'
             and node.inputsAt(0).node().s('name') == 'backbone']
    if len(calls) != 1:
        raise RuntimeError('Unexpected Gen1 backbone graph')
    call = calls[0]
    with graph.insert_point_guard(call):
        key = graph.insertConstant(owner)
    node = graph.create('vision_shield_plate::backbone', [call.inputsAt(1), key], 5)
    node.insertBefore(call)
    packed = graph.create('prim::TupleConstruct', list(node.outputs()), 1)
    packed.output().setType(call.output().type())
    packed.insertBefore(call)
    call.output().replaceAllUsesWith(packed.output())
    call.destroy()
    torch._C._jit_pass_inline(graph)
    graph.lint()
    return graph


class GraphedPlate:
    def __init__(self, model):
        from vision_shield.detection.egoblur import _async_device_moves

        self.model, self.backbone = model.model, model.model.model.backbone
        self.cache = {}
        _owners[id(self)] = self
        # shortcut: fixed Gen1 wrapper; graph guards must be reviewed when changing model exports.
        graph = replace_backbone(self.model.forward.graph.copy(), id(self))
        self.forward = _async_device_moves(model, graph=graph)

    def __call__(self, pixels):
        return self.forward(self.model, pixels)

    def replay(self, pixels):
        key = (tuple(pixels.shape), pixels.dtype, pixels.device)
        if key not in self.cache:
            # Keep the startup/local and full-frame shapes; other shapes retain the exact eager path.
            if len(self.cache) == 2:
                return self.backbone(pixels)
            static = torch.empty_like(pixels)
            static.copy_(pixels)
            stream = torch.cuda.Stream(device=pixels.device)
            stream.wait_stream(torch.cuda.current_stream(pixels.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.backbone(static)
            torch.cuda.current_stream(pixels.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode='thread_local'):
                output = self.backbone(static)
            self.cache[key] = static, graph, output
        static, graph, output = self.cache[key]
        static.copy_(pixels)
        graph.replay()
        # Fresh outputs prevent a later replay or downstream in-place work from changing older results.
        return tuple(value.clone() for value in output)
