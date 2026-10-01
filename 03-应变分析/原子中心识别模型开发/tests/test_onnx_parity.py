from pathlib import Path

import numpy as np
import pytest


@pytest.mark.ml
def test_pytorch_export_matches_onnxruntime(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")

    class TinyModel(torch.nn.Module):
        def forward(self, value: object) -> object:
            return value * 2.0 + 1.0

    model = TinyModel().eval()
    input_tensor = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    model_path = tmp_path / "tiny.onnx"
    torch.onnx.export(
        model,
        input_tensor,
        model_path,
        input_names=["image"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    expected = model(input_tensor).detach().cpu().numpy()
    session = onnxruntime.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    actual = session.run(["output"], {"image": input_tensor.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
