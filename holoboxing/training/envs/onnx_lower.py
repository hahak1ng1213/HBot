"""Batch specialization of the pinned public ONNX, without changing weights."""

from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from holoboxing.evaluation.models import frozen_manifest
from holoboxing.evaluation.runtime.model_contract import sha256


class TrainingBoundSession:
    """Torch-owned ORT buffers, independent of the Warp evaluation runtime."""

    def __init__(self, session, device):
        self.device = torch.device(str(device))
        self.session = session
        self.inputs, self.outputs = {}, {}
        self.binding = session.io_binding()
        for nodes, buffers, bind in (
            (session.get_inputs(), self.inputs, self.binding.bind_input),
            (session.get_outputs(), self.outputs, self.binding.bind_output),
        ):
            for node in nodes:
                shape = tuple(node.shape)
                integer = "int64" in node.type
                buffer = torch.zeros(
                    shape,
                    device=self.device,
                    dtype=torch.int64 if integer else torch.float32,
                )
                buffers[node.name] = buffer
                bind(
                    node.name,
                    self.device.type,
                    self.device.index or 0,
                    np.int64 if integer else np.float32,
                    shape,
                    buffer.data_ptr(),
                )

    def run(self):
        self.binding.synchronize_inputs()
        self.session.run_with_iobinding(self.binding)
        self.binding.synchronize_outputs()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return self.outputs


def training_session(path, device):
    import onnxruntime as ort

    from holoboxing.evaluation.runtime.ordered_topk import execution_source

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.use_deterministic_compute = True
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA policy device is unavailable")
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        provider = "CUDAExecutionProvider"
        if provider not in ort.get_available_providers():
            raise RuntimeError(
                "Install a compatible onnxruntime-gpu for CUDA inference"
            )
        providers = [
            (
                provider,
                {
                    "device_id": device.index or 0,
                    "user_compute_stream": str(
                        torch.cuda.current_stream(device).cuda_stream
                    ),
                    "arena_extend_strategy": "kSameAsRequested",
                    "use_tf32": "0",
                },
            )
        ]
        options.add_session_config_entry(
            "session.disable_cpu_ep_fallback", "1"
        )
    elif device.type == "cpu":
        provider = "CPUExecutionProvider"
        providers = [provider]
    else:
        raise ValueError("Choose cpu or cuda for policy inference")
    session = ort.InferenceSession(
        execution_source(path), sess_options=options, providers=providers
    )
    session.disable_fallback()
    if session.get_providers()[0] != provider:
        raise RuntimeError(f"Requested {provider} was not activated")
    return session


def specialize_graph(path, batch_size):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    manifest = frozen_manifest()
    expected_model = manifest["files"][manifest["model"]]
    if sha256(path) != expected_model["sha256"]:
        raise ValueError(
            "Only the pinned public lower ONNX can be specialized"
        )
    graph = onnx.load(path)
    # These constants are the batch-bearing reshapes in this exact release.
    shapes = {
        10: (1, 1, 8, 64),
        11: (1, 1, 2, 4, 64),
        30: (1, 1, 1, 1),
        31: (1, 1, 4, 64),
        40: (1, 4, 2, 32, 64),
        43: (1, 8, 32, 64),
        45: (1, 4, 2, 32, 64),
        48: (1, 8, 32, 64),
        55: (1, 1, -1),
        58: (1, 512),
        59: (1, 2),
        60: (1, 2),
        62: (1, 2, 512),
        63: (1, 2, 512),
        66: (2, 512),
        67: (2,),
        79: (1, 1, 512),
    }
    remaining = set(shapes)
    for node in graph.graph.node:
        for index, expected in shapes.items():
            if node.name != f"/actor_module/Constant_{index}":
                continue
            actual = numpy_helper.to_array(node.attribute[0].t)
            if tuple(actual) != expected:
                raise ValueError("Frozen V2 specialization constant changed")
            updated = np.asarray(expected, dtype=np.int64)
            updated[0] *= batch_size
            node.attribute[0].t.CopyFrom(numpy_helper.from_array(updated))
            remaining.remove(index)
    if remaining:
        raise ValueError("Frozen V2 specialization nodes are missing")
    for value in (*graph.graph.input, *graph.graph.output):
        axis = (
            2 if value.name in ("past_key_values", "present_key_values") else 0
        )
        value.type.tensor_type.shape.dim[axis].dim_value = batch_size
    del graph.graph.value_info[:]
    onnx.checker.check_model(graph)
    return graph


class OnnxLower:
    """Shared inference; recurrent state belongs to each controller."""

    def __init__(
        self, path, batch_size, device, directory, inference_batch_size=128
    ):
        if any(
            type(value) is not int or value < 1
            for value in (batch_size, inference_batch_size)
        ):
            raise ValueError("Batch sizes must be positive integers")
        self.device = torch.device(str(device))
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.batch_size = batch_size
        # Keep static ONNX batches without restricting the environment count.
        self.chunk_size = next(
            size
            for size in range(min(batch_size, inference_batch_size), 0, -1)
            if batch_size % size == 0
        )
        graph = specialize_graph(path, self.chunk_size)
        self.metadata = {item.key: item.value for item in graph.metadata_props}
        target = Path(directory) / f"lower_batch_{self.chunk_size}.onnx"
        target.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(graph, target)
        del graph
        self.session = training_session(target, self.device)
        expected = {
            "obs": [self.chunk_size, 1378],
            "past_key_values": [1, 2, self.chunk_size, 32, 4, 64],
            "step_idx": [self.chunk_size],
        }
        if {n.name: n.shape for n in self.session.get_inputs()} != expected:
            raise ValueError("Public ONNX input contract differs")
        outputs = {n.name: n.shape for n in self.session.get_outputs()}
        if (
            outputs.get("actions") != [self.chunk_size, 29]
            or outputs.get("present_key_values") != expected["past_key_values"]
        ):
            raise ValueError("Public ONNX output contract differs")
        self.bound = TrainingBoundSession(self.session, self.device)

    @torch.inference_mode()
    def __call__(self, observation, cache, step):
        actions = torch.empty((self.batch_size, 29), device=self.device)
        present = torch.empty_like(cache)
        for start in range(0, self.batch_size, self.chunk_size):
            rows = slice(start, start + self.chunk_size)
            self.bound.inputs["obs"].copy_(observation[rows])
            self.bound.inputs["past_key_values"].copy_(cache[:, :, rows])
            self.bound.inputs["step_idx"].copy_(step[rows])
            # ORT may use its own stream (CUDA stream 0 is not an external
            # stream). Finish input writes and the previous output reads before
            # reusing bound buffers; the shared runner fences output writes.
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            output = self.bound.run()
            actions[rows].copy_(output["actions"])
            present[:, :, rows].copy_(output["present_key_values"])
        return actions, present
