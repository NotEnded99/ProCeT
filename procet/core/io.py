"""Model export and full-LBP verification wrappers.

Thin wrappers around ``lbp_neural_cbf.cbf.verify_cbf`` and ``torch.onnx``.
"""

import torch

from lbp_neural_cbf.cbf.verify_cbf import verify_cbf


def pytorch_to_onnx(model, onnx_path, input_dim=2):
    """Export a PyTorch barrier network to ONNX (opset 14, dynamic batch)."""
    device = next(model.parameters()).device
    model.eval()
    dummy_input = torch.randn(1, input_dim, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def verify_model(model_path, dynamics_model, max_depth=13):
    """Verify a barrier network with full LBP (McCormick).

    Args:
        model_path: Path to a saved ``.pth`` barrier network.
        dynamics_model: CBF dynamics system instance.
        max_depth: Maximum simplicial splitting depth.

    Returns:
        Raw result dict from ``verify_cbf`` — keys include ``V_safe``,
        ``V_unsafe``, ``F_h_positive_in_unsafe``, ``F_safe_cbf_violation``,
        ``F_depth_limit_reached_unsafe``, ``F_depth_limit_reached_safe``,
        ``F_unsafe_cannot_split``.
    """
    return verify_cbf(
        dynamics_model,
        barrier_model_path=model_path,
        visualize=False,
        use_gpu=True,
        batch_size=512,
        executor_type="single",
        region_type="simplicial",
        max_depth=max_depth,
    )
