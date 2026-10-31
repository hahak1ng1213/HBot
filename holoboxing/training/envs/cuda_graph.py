"""Fixed-shape, inference-only CUDA graph for pure tensor preprocessing."""

import torch


class InferenceGraph:
    """Borrowed outputs last until next call; no recurrent state capture."""

    def __init__(self, function):
        self.function = function
        self.graph = None

    @torch.no_grad()
    def __call__(self, value):
        if value.device.type != "cuda":
            return self.function(value)
        if value.requires_grad:
            raise ValueError("Inference graph cannot consume gradient inputs")
        with torch.cuda.device(value.device):
            stream = torch.cuda.current_stream(value.device)
            if self.graph is None:
                self.input = value.clone()
                self.stream = stream
                warmup = torch.cuda.Stream(device=value.device)
                warmup.wait_stream(stream)
                with torch.cuda.stream(warmup):
                    for _ in range(3):
                        self.function(self.input)
                stream.wait_stream(warmup)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=warmup):
                    self.output = self.function(self.input)
                stream.wait_stream(warmup)
            if (
                value.shape != self.input.shape
                or value.dtype != self.input.dtype
                or value.device != self.input.device
                or stream != self.stream
            ):
                raise ValueError(
                    "Inference graph input shape, device or stream changed"
                )
            self.input.copy_(value)
            self.graph.replay()
            return self.output
