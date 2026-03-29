$(head -60 experiments/icgar_repair_backup.py)
"""
ICGAR Repair: Iterative Certificate-Gradient Aligned Refinement
"""
import sys
from pathlib import Path
cwd = str(Path.cwd())
if cwd not in sys.path:
    sys.path.insert(0, cwd)

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from lbp_neural_cbf.cbf.verify_cbf import verify_cbf
from lbp_neural_cbf.cbf.network import BarrierNN, empirical_cbf_validation
from lbp_neural_cbf.cbf.fossil_dynamics import Barrier1System, Barrier2System, Barrier3System, Barrier4System
from lbp_neural_cbf.certification_results import SampleResultUNSAT
from lbp_neural_cbf.translators import TorchTranslator


def get_system_model(system_type, alpha=1.0):
    """Get dynamics model for a given system type."""
    system_type = system_type.lower()
    if system_type == "barr1":
        return Barrier1System(alpha=alpha)
    elif system_type == "barr2":
        return Barrier2System(alpha=alpha)
    elif system_type == "barr3":
        return Barrier3System(alpha=alpha)
    elif system_type == "barr4":
        return Barrier4System(alpha=alpha)
    else:
        raise ValueError("Unknown system type: " + system_type)


def collect_counterexamples_from_results(results_list):
    """Extract counterexamples from verification results."""
    counterexamples = []
    
    for item in results_list:
        if hasattr(item, 'sample'):
            sample = item.sample
            if hasattr(sample, 'isunsat') and sample.isunsat():
                if isinstance(item, SampleResultUNSAT):
                    if item.hascounterexamples():
                        for cex in item.counterexamples():
                            if isinstance(cex, np.ndarray):
                                if len(cex.shape) > 0:
                                    if cex.shape[1] > 0:
                                        counterexamples.append(cex)
                elif hasattr(sample, 'center_point'):
                    counterexamples.append(sample.center_point)
                elif hasattr(sample, 'vertices') and len(sample.vertices) > 0:
                    counterexamples.append(sample.vertices[0])
    
    return counterexamples


class SimpleRepair:
    """Simple gradient-based repair."""
    
    def __init__(self, dynamics_model, barrier_net, device, lr=1e-3, epochs=100, lambda_repair=10.0, lambda_l2=1e-5):
        self.dynamics_model = dynamics_model
        self.barrier_net = barrier_net
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.lambda_repair = lambda_repair
        self.lambda_l2 = lambda_l2
        self.initial_params = [p.clone().detach() for p in self.barrier_net.parameters()]
        self.optimizer = optim.AdamW(self.barrier_net.parameters(), lr=lr, weight_decay=lambda_l2)
        self.history = []

    def compute_repair_loss(self, cex_tensor):
        if cex_tensor.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)
        h_values = self.barrier_net(cex_tensor).squeeze()
        h_violation = F.relu(-h_values).mean()
        repair_loss = h_violation
        l2_loss = torch.tensor(0.0, device=self.device)
        for param, init_param in zip(self.barrier_net.parameters(), self.initial_params):
            l2_loss += torch.norm(param - init_param) ** 2
        return self.lambda_repair * repair_loss + self.lambda_l2 * l2_loss

    def run_repair(self, initial_results):
        counterexamples = collect_counterexamples_from_results(initial_results)
        initial_cex = len(counterexamples)
        
        print("\n" + "=" * 60)
        print("ICGAR REPAIR PHASE")
        print("=" * 60)
        print("Initial counterexamples: " + str(initial_cex))
        print("Max iterations: " + str(self.epochs))
        print("Learning rate: " + str(self.lr))
        
        if initial_cex == 0:
            print("No counterexamples - repair not needed!")
            return {"success": True, "iterations": 0, "initial_cex": 0}
        
        print("Training on " + str(initial_cex) + " counterexamples")
        cex_tensor = torch.tensor(np.array(counterexamples), dtype=torch.float32, device=self.device)
        
        pbar = tqdm(range(self.epochs), desc="Repairing", unit="iter")
        
        for iteration in pbar:
            self.optimizer.zero_grad()
            loss = self.compute_repair_loss(cex_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.barrier_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            self.history.append({"iteration": iteration, "loss": loss.item()})
            
            if (iteration + 1) % 20 == 0 or (iteration + 1) == self.epochs:
                with torch.no_grad():
                    ver_metrics, _ = empirical_cbf_validation(
                        self.barrier_net,
                        self.dynamics_model,
                        num_samples=5000,
                        alpha=self.dynamics_model.alpha,
                    )
                    pbar.set_postfix({
                        "loss": f"{loss.item():.6f}",
                        "violations": f"{ver_metrics['violation_rate']:.2f}%",
                        "score": f"{ver_metrics['validity_score']:.2f}",
                    })
                    self.history[-1].update({
                        "violation_rate": ver_metrics["violation_rate"],
                        "validity_score": ver_metrics["validity_score"],
                    })
        
        pbar.close()
        
        final_ver, _ = empirical_cbf_validation(
            self.barrier_net,
            self.dynamics_model,
            num_samples=10000,
            alpha=self.dynamics_model.alpha,
        )
        
        print("\n" + "=" * 60)
        print("REPAIR COMPLETE")
        print("=" * 60)
        print("Final violation rate: " + str(final_ver['violation_rate']) + "%")
        print("Final validity score: " + str(final_ver['validity_score']))
        
        return {
            "success": True,
            "iterations": self.epochs,
            "initial_cex": initial_cex,
            "history": self.history,
            "final_metrics": final_ver,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ICGAR Repair for Neural CBFs")
    parser.add_argument("--system-type", type=str, default="barr3", help="System type (default: barr3)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to input ONNX model")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max-depth", type=int, default=13, help="Max verification depth (default: 13)")
    parser.add_argument("--repair-epochs", type=int, default=100, help="Number of repair iterations (default: 100)")
    parser.add_argument("--repair-lr", type=float, default=1e-3, help="Repair learning rate (default 1e-3)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ICGAR Repair: Simple Empirical Repair")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))
    
    dynamics_model = get_system_model(args.system_type)
    print("System: " + dynamics_model.system_name)
    
    if args.model_path is None:
        model_path = "data/mine_models_relu/" + dynamics_model.system_name + "_cbf.onnx"
        pth_path = "data/mine_models_relu/" + dynamics_model.system_name + "_cbf.pth"
    else:
        pth_path = args.model_path.replace(".onnx", ".pth")
    
    if args.output_dir is None:
        output_dir = "data/icgar_repaired_" + args.system_type
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nLoading model: " + model_path)
    
    barrier_net = BarrierNN(
        input_size=dynamics_model.input_dim,
        hidden_sizes=dynamics_model.hidden_sizes,
        device=device,
        activation_fnc=dynamics_model.activation_fnc,
    )
    barrier_net.load_state_dict(torch.load(pth_path, map_location=device, weights_only=False))
    barrier_net.eval()
    
    print("\n" + "-" * 60)
    print("INITIAL VERIFICATION")
    print("-" * 60)
    initial_results = verify_cbf(
        dynamics_model,
        barrier_model_path=model_path,
        executor_type="single",
        region_type="simplicial",
        use_gpu=(device.type == "cuda"),
        use_wandb=False,
        max_depth=args.max_depth,
    )
    
    initial_pass_rate = initial_results["certified_percentage"]
    print("Initial pass rate: " + str(initial_pass_rate) + "%")
    
    repair = SimpleRepair(
        dynamics_model=dynamics_model,
        barrier_net=barrier_net,
        device=device,
        lr=args.repair_lr,
        epochs=args.repair_epochs,
    )
    
    repair_result = repair.run_repair(initial_results.get("regions", []))
    
    repaired_pth_path = os.path.join(output_dir, dynamics_model.system_name + "_cbf_repaired.pth")
    repaired_onnx_path = os.path.join(output_dir, dynamics_model.system_name + "_cbf_repaired.onnx")
    
    print("\nSaving repaired model to: " + repaired_pth_path)
    torch.save(barrier_net.state_dict(), repaired_pth_path)
    
    print("Saving ONNX model to: " + repaired_onnx_path)
    dummy_input = torch.randn(1, dynamics_model.input_dim, device=device)
    torch.onnx.export(
        barrier_net,
        dummy_input,
        repaired_onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    
    history_path = os.path.join(output_dir, "repair_history.json")
    with open(history_path, "w") as f:
        json.dump(repair_result, f, indent=2, default=str)
    print("Repair history saved to: " + history_path)
    
    print("\n" + "-" * 60)
    print("FINAL VERIFICATION")
    print("-" * 60)
    print("Running final verification on repaired model...")
    final_results = verify_cbf(
        dynamics_model,
        barrier_model_path=repaired_onnx_path,
        executor_type="single",
        region_type="simplicial",
        use_gpu=(device.type == "cuda"),
        use_wandb=False,
        max_depth=args.max_depth,
    )
    
    final_pass_rate = final_results["certified_percentage"]
    improvement = final_pass_rate - initial_pass_rate
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("System: " + args.system_type)
    print("Initial pass rate: " + str(initial_pass_rate) + "%")
    print("Final pass rate: " + str(final_pass_rate) + "%")
    print("Improvement: " + str(improvement) + "%")
    
    summary = {
        "system_type": args.system_type,
        "initial_pass_rate": initial_pass_rate,
        "final_pass_rate": final_pass_rate,
        "improvement": improvement,
        "repair_epochs": args.repair_epochs,
        "repair_lr": args.repair_lr,
        "max_depth": args.max_depth,
    }
    
    summary_path = os.path.join(output_dir, "repair_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved to: " + summary_path)
    
    return repair_result


if __name__ == "__main__":
    main()
