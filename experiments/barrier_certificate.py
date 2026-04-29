import sys
from pathlib import Path
cwd = str(Path.cwd())
if cwd not in sys.path:
    sys.path.insert(0, cwd)

import multiprocessing
import os
import json

from lbp_neural_cbf.cbf.train_cbf import train_cbf
from lbp_neural_cbf.cbf.verify_cbf import verify_cbf
from lbp_neural_cbf.cbf.cbf_dynamics import Simple2DSystem, CartPoleSystem, RendezvousDockingSystem
from lbp_neural_cbf.cbf.fossil_dynamics import Barrier1System, Barrier2System, Barrier3System, Barrier4System, HighOrd2System, HighOrd4System, HighOrd6System, HighOrd8System, PlanarQuadrotorSystem
from lbp_neural_cbf.visualization.cbf_plotter import create_cbf_verification_plotter

def main(system_type="barr1", train=True, verify=False, alpha=1.0, region_type="simplicial", executor_type="single", max_depth=None, activation=None, hidden_sizes=None, save_path=None):
    """
    Main script for training and verifying neural control barrier functions.

    Args:
        system_type: Type of dynamical system ("simple2d", "barr1", "barr2", "barr3", "barr4", "hiord2", "hiord4", "cartpole", "hiord6", "hiord8", "rendezvousdocking", "planarquad", "LeakyRelu")
        train: Whether to train the CBF
        verify: Whether to verify the CBF
        alpha: Alpha parameter for the CBF
        region_type: Type of regions to use for verification ("hyperrectangular" or "simplicial")
        executor_type: Type of executor ("single", "multi-thread", or "multi-process")
        max_depth: Maximum depth for region splitting (None for unlimited)
    """
    
    print("="*60)
    print("NEURAL CONTROL BARRIER FUNCTION EXPERIMENT")
    print("="*60)

    batch_size = 512  # Default batch size

    # Select dynamical system
    if system_type.lower() == "simple2d":
        dynamics_model = Simple2DSystem(alpha=alpha)
        print("Using Simple 2D System (with constant control: g(x) = I)")
    elif system_type.lower() == "barr1":
        dynamics_model = Barrier1System(alpha=alpha)
        batch_size = 8  # Smaller batch size for large network
        print("Using FOSSIL Barrier 1 System")
    elif system_type.lower() == "barr2":
        dynamics_model = Barrier2System(alpha=alpha)
        print("Using FOSSIL Barrier 2 System")
    elif system_type.lower() == "barr3":
        dynamics_model = Barrier3System(alpha=alpha)
        print("Using FOSSIL Barrier 3 System")
    elif system_type.lower() == "barr4":
        dynamics_model = Barrier4System(alpha=alpha)
        batch_size = 256  # Smaller batch size for higher-dimensional system
        print("Using FOSSIL Barrier 4 System")
    elif system_type.lower() == "hiord2":
        dynamics_model = HighOrd2System(alpha=alpha)
        print("Using High-Order 2D System")
    elif system_type.lower() == "hiord4":
        dynamics_model = HighOrd4System(alpha=alpha)
        batch_size = 128  # Smaller batch size for higher-dimensional system
        print("Using High-Order 4D System")
    elif system_type.lower() == "cartpole":
        dynamics_model = CartPoleSystem(alpha=alpha)
        batch_size = 128  # Smaller batch size for higher-dimensional system
        print("Using CartPole System")
    elif system_type.lower() == "hiord6":
        dynamics_model = HighOrd6System(alpha=alpha)
        batch_size = 256  # Smaller batch size for higher-dimensional system
        print("Using High-Order 6D System")
    elif system_type.lower() == "hiord8":
        dynamics_model = HighOrd8System(alpha=alpha)
        batch_size = 256  # Smaller batch size for higher-dimensional system
        print("Using High-Order 8D System")
    elif system_type.lower() == "rendezvousdocking":
        dynamics_model = RendezvousDockingSystem(alpha=alpha)
        batch_size = 64  # Smaller batch size for higher-dimensional system
        print("Using Rendezvous Docking System")
    elif system_type.lower() == "planarquad":
        dynamics_model = PlanarQuadrotorSystem(alpha=alpha)
        batch_size = 256  # Smaller batch size for 6D system
        print("Using Planar Quadrotor System")
    else:
        raise ValueError(f"Unknown system type: {system_type}")
    
    print(f"System parameters:")
    print(f"  Input dimension: {dynamics_model.input_dim}")
    print(f"  Control dimension: {dynamics_model.control_dim}")
    if dynamics_model.control_dim > 0:
        print(f"  Control bounds: u ∈ [{dynamics_model.u_min}, {dynamics_model.u_max}]")
    print(f"  Safe set: {dynamics_model.safe_set}")
    print(f"  Alpha parameter: {dynamics_model.alpha}")
    print(f"  Input domain: {dynamics_model.input_domain}")
    
    barrier_net = None
    
        # Override activation function and save paths if specified
    if activation is not None:
        valid_activations = ["Tanh", "Relu", "Sigmoid", "LeakyRelu"]
        if activation not in valid_activations:
            raise ValueError(f"Invalid activation function: {activation}. Must be one of {valid_activations}")
        dynamics_model.activation_fnc = activation
        print(f"  Using activation function: {activation} (overriding default)")

    # Override hidden sizes if specified (e.g. "64,128,64" -> [64, 128, 64])
    if hidden_sizes is not None:
        parsed = [int(x.strip()) for x in hidden_sizes.split(",")]
        dynamics_model.hidden_sizes = parsed
        print(f"  Using hidden sizes: {parsed} (overriding default)")

    if train:
        print("\n" + "-"*40)
        print("TRAINING PHASE")
        print("-"*40)
        
        # Default/fallback training parameters
        training_params = {}

        # Try to load best training params for this system if available
        system_key = dynamics_model.system_name
        
        # best_params_path = os.path.join('data', 'cbf_best_params', f"{system_key}_best_params.json")
        # if os.path.exists(best_params_path):
        #     print(f"Found best-params JSON for {system_key}: {best_params_path}")
        #     with open(best_params_path, 'r') as f:
        #         payload = json.load(f)
        #     best_training_params = payload.get('training_params', {})
        #     if best_training_params:
        #         print("Using best training parameters from JSON.")
        #         training_params = best_training_params
        #         training_params['use_wandb'] = True  # Enable wandb logging for standard training
        #         training_params['wandb_viz_freq'] = 500
        #     else:
        #         print("Best-params JSON present but missing 'training_params' field; using defaults.")
        # else:
        #     print("No best-params JSON found; using default training parameters.")
        print("No best-params JSON found; using default training parameters.")
        
        print("Training parameters:")
        for key, value in training_params.items():
            print(f"  {key}: {value}")
        

        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            save_path_torch = os.path.join(save_path, f"{system_key}_cbf.pth")
            save_path_onnx = os.path.join(save_path, f"{system_key}_cbf.onnx")
            print(f"  Torch model will be saved to: {save_path_torch}")
            print(f"  ONNX model will be saved to: {save_path_onnx}")
        # Train the CBF
        barrier_net = train_cbf(
            dynamics_model,
            save_path_torch=save_path_torch,
            save_path_onnx=save_path_onnx,
            num_epochs = 8000, 
            **training_params
        )
    
    if verify:
        print("\n" + "-"*40)
        print("VERIFICATION PHASE")
        print("-"*40)

        # Define path to the ONNX model for verification
        # network_path = f"data/mine_models_relu/{dynamics_model.system_name}_cbf.onnx"
        # network_path = f"data/author_models/{dynamics_model.system_name}_cbf.onnx"
        # if activation == "Tanh":
        #     network_path = f"/data/mzm/mzm_Verification/verification-of-neural-cbf-mzm4/data/New_models_Hard_Tanh_v1/{dynamics_model.system_name}_cbf.onnx"
        # elif activation == "Relu":
        #     network_path = f"/data/mzm/mzm_Verification/verification-of-neural-cbf-mzm4/data/New_models_Hard_Relu_v1/{dynamics_model.system_name}_cbf.onnx"
        # elif activation == "Sigmoid":
        #     network_path = f"/data/mzm/mzm_Verification/verification-of-neural-cbf-mzm4/data/New_models_Hard_Sigmoid_v1/{dynamics_model.system_name}_cbf.onnx"
        # else:
        #     raise ValueError(f"Unsupported activation function for verification: {activation}")
        
        if activation == "Tanh":
            network_path = f"/data/mzm/Repair_NCBF/data/New_models_Hard_Tanh_v1/{dynamics_model.system_name}_cbf.onnx"
        elif activation == "Relu":
            network_path = f"/data/mzm/Repair_NCBF/data/New_models_Hard_Relu_v1/{dynamics_model.system_name}_cbf.onnx"
        elif activation == "Sigmoid":
            network_path = f"/data/mzm/Repair_NCBF/data/New_models_Hard_Sigmoid_v1/{dynamics_model.system_name}_cbf.onnx"
        elif activation == "LeakyRelu":
            network_path = f"/data/mzm/Repair_NCBF/data/New_models_Hard_LeakyRelu_v1/{dynamics_model.system_name}_cbf.onnx"
        else:
            raise ValueError(f"Unsupported activation function for verification: {activation}")
    
        # Verification parameters
        verification_params = {
            'executor_type': executor_type,  # Type of executor (single, multi-thread, or multi-process)
            'region_type': region_type,  # Use specified region type for verification
            'max_depth': max_depth  # Maximum depth for region splitting
        }
        
        print("Verification parameters:")
        for key, value in verification_params.items():
            print(f"  {key}: {value}")
        
        print(f"\nUsing {region_type} regions and {executor_type} executor for modular CBF verification...")
        
        # Run verification
        performance_test = True
        results = verify_cbf(
            dynamics_model, 
            network_path,
            visualize=False if performance_test else True,  # Enable live visualization for 2D systems
            use_wandb=False if performance_test else True,  # Enable wandb logging for verification phase
            use_gpu=True,   # Use GPU for faster verification
            batch_size=batch_size,
            save_verification_regions = True,  # Save verification regions for later analysis
            **verification_params
        )
        
        # Additional analysis
        if results['uncertified_percentage'] == 0.0:
            print("\n✅ CBF verification PASSED - No counterexamples found!")
        else:
            print(f"\n⚠️  CBF verification: uncertified percentage is {results['uncertified_percentage']:.4f}%")
        
            print("\nCounterexample analysis:")
            cex_types = {}
            total_counterexamples = 0
            
            for region in results['regions']:
                if region.isunsat() and region.hascounterexamples():
                    counterexamples = region.counterexamples()
                    total_counterexamples += len(counterexamples)
                    
                    for cex in counterexamples:
                        if isinstance(cex, dict):
                            cex_type = cex.get('type', 'unknown')
                        else:
                            # Handle case where counterexample is not a dict
                            cex_type = 'violation'
                        cex_types[cex_type] = cex_types.get(cex_type, 0) + 1
            
            print(f"  Total counterexamples found: {total_counterexamples}")
            
            for cex_type, count in cex_types.items():
                print(f"  {cex_type}: {count}")
            
            # Show details for different types
            if 'gradient_error' in cex_types:
                print("\n  Note: Gradient errors may indicate numerical issues at boundary regions.")
                print("        These don't necessarily mean the CBF is invalid.")
            
            if 'h_negative' in cex_types:
                print("\n  ⚠️  h_negative errors indicate serious CBF violations!")
            
            if 'cbf_violation' in cex_types:
                print("\n  ⚠️  CBF condition violations indicate the barrier may not be valid!")
                
            if 'violation' in cex_types:
                print("\n  ⚠️  CBF violations found - the barrier function may not satisfy the CBF condition!")

if __name__ == "__main__":
    import argparse

    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This must be done before any CUDA operations
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Neural Control Barrier Function Experiment")
    parser.add_argument("--system-type", type=str, default="barr1",
                       help="Type of dynamical system (default: barr1)")
    parser.add_argument("--train", action="store_true", 
                       help="Whether to train the CBF")
    parser.add_argument("--verify", action="store_true", 
                       help="Whether to verify the CBF (default: True)")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Alpha parameter for the CBF (default: 1.0)")
    parser.add_argument("--region-type", type=str, default="simplicial",
                       choices=["hyperrectangular", "simplicial"],
                       help="Type of regions to use for verification (default: simplicial)")
    parser.add_argument("--executor-type", type=str, default="single",
                       choices=["single", "multi-thread", "multi-process"],
                       help="Type of executor (default: single)")
    parser.add_argument("--max-depth", type=int, default=15,
                       help="Maximum depth for region splitting (None for unlimited)")
    
    parser.add_argument("--activation", type=str, default="Tanh", choices=["Tanh", "Relu", "Sigmoid", "LeakyRelu"],
                       help="Activation function for the barrier network (overrides system default)")
    parser.add_argument("--hidden-sizes", type=str, default='32,64,32',
                       help="Hidden layer sizes, e.g. '32,64,32' (overrides system default)")
    parser.add_argument("--save-path", type=str, default="data/New_models_Hard_Tanh")

    args = parser.parse_args()

    # Run experiment with parsed parameters
    main(
        system_type=args.system_type,
        train=args.train,
        verify=args.verify,
        alpha=args.alpha,
        region_type=args.region_type,
        executor_type=args.executor_type,
        max_depth=args.max_depth,
        activation=args.activation,
        hidden_sizes=args.hidden_sizes,
        save_path=args.save_path,
    )
