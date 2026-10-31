"""CUDA preprocessing parity; these are not learned boxing quality tests."""

import pytest
import torch

from holoboxing.training.envs.cuda_graph import InferenceGraph
from holomotion.src.motion_tracking.link_fk import SharedLinkFK


@pytest.mark.parametrize("batch", [8, 32])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fk_graph_matches_eager_after_input_change_and_reset(
    tmp_path, batch, device
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    path = tmp_path / "robot.xml"
    path.write_text("""<mujoco><worldbody><body name="pelvis">
      <body name="hip" pos="0 0 -.2"><joint name="hip_joint" axis="0 1 0"/>
      <body name="foot" pos=".1 0 -.3"><joint name="ankle_joint" axis="1 0 0"/>
      </body></body></body></worldbody><actuator>
      <motor joint="hip_joint"/><motor joint="ankle_joint"/>
      </actuator></mujoco>""")
    fk = SharedLinkFK(str(path), device)
    graph = InferenceGraph(fk)
    with torch.no_grad():
        for step in range(12):
            # Strided input and partial resets must update captured inputs.
            value = torch.randn(batch, 4, device=device)[:, ::2]
            if step % 3 == 0:
                value[0] = 0
            expected = fk(value)
            actual = graph(value)
            for a, b in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        if device.startswith("cuda"):
            with pytest.raises(ValueError, match="shape"):
                graph(torch.zeros(batch + 1, 2, device=device))


def test_cpu_preprocessing_uses_eager():
    graph = InferenceGraph(lambda x: (x.square(), x + 1))
    x = torch.tensor([2.0, 3.0])
    torch.testing.assert_close(graph(x)[0], x.square())
    assert graph.graph is None
