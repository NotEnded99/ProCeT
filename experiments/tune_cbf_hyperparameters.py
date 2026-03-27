"""
Hyperparameter tuning for Control Barrier Function training using Weights & Biases sweeps.

This script uses W&B sweeps to automatically explore hyperparameter configurations
and find the best settings for training neural control barrier functions.

The optimization metric is based on CBF validity - we use FAST EMPIRICAL VALIDATION
to check CBF conditions on 50K sampled points. This is much faster than formal verification
while still providing a good assessment of CBF validity by counting violations.
"""

import multiprocessing
import os
import json
import wandb
import torch
import math
import numpy as np
from lbp_neural_cbf.cbf.train_cbf import train_cbf
from lbp_neural_cbf.cbf.cbf_dynamics import Simple2DSystem, CartPoleSystem, RendezvousDockingSystem
from lbp_neural_cbf.cbf.fossil_dynamics import Barrier1System, Barrier2System, Barrier3System, Barrier4System, HighOrd2System, HighOrd4System, HighOrd6System, HighOrd8System
from lbp_neural_cbf.translators import TorchTranslator
from lbp_neural_cbf.cbf.network import empirical_cbf_validation

# Track best hyperparameters per system in-memory during a sweep session
BEST_BY_SYSTEM = {}


def _is_better_result(new_ver, old_ver):
    """Compare validation results: prefer zero violations; then larger invariant set.
    Fallback to higher validity_score; ties broken by lower violation_rate.
    """
    if old_ver is None:
        return True

    new_zero = new_ver.get('total_violations') == 0
    old_zero = old_ver.get('total_violations') == 0
    if new_zero and not old_zero:
        return True
    if old_zero and not new_zero:
        return False
    if new_zero and old_zero:
        # Maximize coverage, then size
        return new_ver.get('invariant_set_coverage') > old_ver.get('invariant_set_coverage')

    # No zero-violation configs: minimize total violations; then maximize validity_score; then minimize violation rate
    if new_ver.get('total_violations') != old_ver.get('total_violations'):
        return new_ver.get('total_violations') < old_ver.get('total_violations')
    if new_ver.get('validity_score') != old_ver.get('validity_score'):
        return new_ver.get('validity_score') > old_ver.get('validity_score')
    return new_ver.get('violation_rate') < old_ver.get('violation_rate')


def _save_best_params(system_type, training_params, validation_results, extra_metrics=None):
    params_dir = os.path.join('data', 'cbf_best_params')
    os.makedirs(params_dir, exist_ok=True)
    out_path = os.path.join(params_dir, f"{system_type}_best_params.json")

    payload = {
        'system_type': system_type,
        'training_params': training_params,
        'validation_results': validation_results,
        'metrics': extra_metrics or {},
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    return out_path


def train_with_config():
    """
    Training function that will be called by W&B sweep agent.
    This function reads hyperparameters from wandb.config and trains the CBF.
    """

    # Initialize wandb run (will be managed by sweep agent)
    run = wandb.init()

    # Get hyperparameters from sweep config
    config = wandb.config

    # Select dynamical system based on config (fossil dynamics)
    system_type = config.get('system_type')
    alpha = config.get('alpha', 1.0)

    st = system_type.lower()
    if st in ("barr1", "barrier1"):
        dynamics_model = Barrier1System(alpha=alpha)
    elif st in ("barr2", "barrier2"):
        dynamics_model = Barrier2System(alpha=alpha)
    elif st in ("barr3", "barrier3"):
        dynamics_model = Barrier3System(alpha=alpha)
    elif st in ("barr4", "barrier4"):
        dynamics_model = Barrier4System(alpha=alpha)
    elif st in ("hiord2", "highord2"):
        dynamics_model = HighOrd2System(alpha=alpha)
    elif st in ("hiord4", "highord4"):
        dynamics_model = HighOrd4System(alpha=alpha)
    elif st in ("hiord6", "highord6"):
        dynamics_model = HighOrd6System(alpha=alpha)
    elif st in ("hiord8", "highord8"):
        dynamics_model = HighOrd8System(alpha=alpha)
    elif st in ("simple2d", "simple_2d"):
        dynamics_model = Simple2DSystem(alpha=alpha)
    elif st in ("cartpole", "cart_pole"):
        dynamics_model = CartPoleSystem(alpha=alpha)
    elif st in ("rendezvous", "rendezvous_docking"):
        dynamics_model = RendezvousDockingSystem(alpha=alpha)
    else:
        raise ValueError(f"Unknown system type for fossil dynamics: {system_type}")

    print(f"\n{'='*60}")
    print(f"W&B Sweep Run: {run.name}")
    print(f"{'='*60}")
    print(f"System: {system_type}")
    print(f"Hyperparameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")

    # Train the CBF with hyperparameters from config
    # Proportions: expects dict with keys 'safe', 'unsafe', 'boundary'.
    proportions = {
        'safe': config.get('prop_safe', 0.1),
        'unsafe': config.get('prop_unsafe', 0.1),
        'boundary': config.get('prop_boundary', 0.5),
    }
    training_params = {
        'learning_rate': getattr(config, 'learning_rate', 1e-3),
        'num_epochs': getattr(config, 'num_epochs', 10000),
        'batch_size': getattr(config, 'batch_size', 32768),
        'data_regen_freq': getattr(config, 'data_regen_freq', 50),
        'proportions': getattr(config, 'proportions', proportions),
        'alpha': getattr(config, 'alpha', 1.0),
        'lambda_safe': getattr(config, 'lambda_safe', 1e-2),
        'lambda_unsafe': getattr(config, 'lambda_unsafe', 10.0),
        'lambda_unsafe_max': getattr(config, 'lambda_unsafe_max', 1.0),
        'lambda_cbf': getattr(config, 'lambda_cbf', 100.0),
        'lambda_bndry': getattr(config, 'lambda_bndry', 0.0),
        'unsafe_margin': getattr(config, 'unsafe_margin', 0.01),
        'safe_margin': getattr(config, 'safe_margin', 0.01),
        'cbf_margin': getattr(config, 'cbf_margin', 0.5),
        'weight_decay': getattr(config, 'weight_decay', 1e-5),
        'min_epochs': getattr(config, 'min_epochs', 1000),
        'curriculum_learning': getattr(config, 'curriculum_learning', True),
        'curriculum_min_epochs': getattr(config, 'curriculum_min_epochs', 1000),
        'save_path_torch': getattr(config, 'save_path_torch', None),
        'save_path_onnx': getattr(config, 'save_path_onnx', None),
        'use_amp': getattr(config, 'use_amp', True),
        'use_wandb': True,
        'wandb_project': getattr(config, 'wandb_project', 'cbf-training'),
        # Disable heavy visualizations during sweeps to save time/bandwidth
        'wandb_viz_freq': 0,
        'wandb_metrics_freq': getattr(config, 'wandb_metrics_freq', 10),
        'validate_during_training_freq': getattr(config, 'validate_during_training_freq', 0),
        'validate_num_samples': getattr(config, 'validate_num_samples', 2000),
    }

    # Train the CBF
    barrier_net = train_cbf(dynamics_model, **training_params)

    # Run empirical validation to assess CBF validity
    print("\n" + "="*60)
    print("Running empirical validation to assess CBF validity...")
    print("="*60)

    validation_results, _ = empirical_cbf_validation(
        barrier_net,
        dynamics_model,
        num_samples=50000,
        alpha=config.alpha
    )
    total_violations = validation_results['total_violations']
    unsafe_set_violations = validation_results['unsafe_set_violations']
    set_invariance_violations = validation_results['set_invariance_violations']
    violation_rate = validation_results['violation_rate']
    unsafe_classification_rate = validation_results['unsafe_classification_rate']
    set_invariance_satisfaction_rate = validation_results['set_invariance_satisfaction_rate']
    invariant_set_size = validation_results['invariant_set_size']
    invariant_set_coverage = validation_results['invariant_set_coverage']
    validity_score = validation_results['validity_score']

    # Log comprehensive metrics to wandb
    metrics_dict = {
        # Primary optimization metric
        'validity_score': validity_score,  # MAXIMIZE this (100+ if no violations)

        # Violation metrics (MINIMIZE these)
        'violation_rate': violation_rate,

        # Success metrics
        'unsafe_classification_rate': unsafe_classification_rate,  # % of unsafe correctly classified
        'set_invariance_satisfaction_rate': set_invariance_satisfaction_rate,  # % of h>=0 satisfying CBF

        # Control invariant set metrics (MAXIMIZE these)
        'invariant_set_coverage': invariant_set_coverage,  # % of safe set covered by h>=0
    }

    # Log to wandb using the current run step to avoid out-of-order step warnings
    current_step = run.step if hasattr(run, 'step') else getattr(wandb.run, 'step', None)
    if current_step is not None:
        wandb.log(metrics_dict, step=current_step)
    else:
        wandb.log(metrics_dict)

    # Also save as summary so they persist and are easily visible in sweep comparison
    for key, value in metrics_dict.items():
        wandb.run.summary[key] = value

    print(f"\n{'='*60}")
    print(f"Final Scores:")
    print(f"  CBF Validity Score: {validity_score:.2f} (>100 is perfect with large invariant set)")
    print(f"  Total Violations: {total_violations} / 50,000 samples")
    print(f"  - Classification violations: {unsafe_set_violations} (h>=0 in unsafe)")
    print(f"  - Set invariance violations: {set_invariance_violations} (CBF fails)")
    print(f"  Control Invariant Set: {invariant_set_size} points ({invariant_set_coverage:.2f}% of safe set)")
    print(f"{'='*60}")

    # Update and persist best parameters per system
    current_best = BEST_BY_SYSTEM.get(system_type)
    current_best_ver = current_best['validation_results'] if current_best else None
    if _is_better_result(validation_results, current_best_ver):
        BEST_BY_SYSTEM[system_type] = {
            'training_params': training_params,
            'validation_results': validation_results,
            'metrics': metrics_dict,
        }
        saved_path = _save_best_params(system_type, training_params, validation_results, metrics_dict)
        print(f"Saved new best params for {system_type} → {saved_path}")

    # Finish the wandb run
    wandb.finish()

    # Return success
    return barrier_net


def create_sweep_config(system_type="barr1", sweep_method="random"):
    """
    Create a W&B sweep configuration for hyperparameter tuning.
    
    Args:
        system_type: Type of dynamical system to tune for
        sweep_method: Sweep method ('grid', 'random', or 'bayes')
        
    Returns:
        Dictionary with sweep configuration
    """

    # Load hyperparameter ranges based on system type
    st = system_type.lower()
    params_path = os.path.join('data', 'cbf_hyperparams_config', f"{st}_cbf_sweep_config.json")

    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            sweep_config = json.load(f)
    else:
        sweep_config = {
            'method': sweep_method,  # Can be 'grid', 'random', or 'bayes'
            'name': f'cbf-{system_type}-validity-sweep',
            'metric': {
                'name': 'validity_score',  # Optimize for CBF validity, not just test accuracy
                'goal': 'maximize'
            },
            'parameters': {
                # System configuration
                'system_type': {
                    'value': system_type
                },
                # Learning rate - log-uniform distribution for exponential search
                "batch_size": {
                    "value": 32768
                },
                "learning_rate": {
                    "distribution": "uniform",
                    "min": 1e-5,
                    "max": 1e-3
                },
                "num_epochs": {
                    "value": 10000
                },
                "min_epochs": {
                    "value": 1000
                },
                "data_regen_freq": {
                    "value": 50
                },
                "alpha": {
                    "value": 1.0
                },
                "weight_decay": {
                    "value": 1e-5
                },
                "curriculum_learning": {
                    "value": "True"
                },
                "curriculum_min_epochs": {
                    "value": 1000
                },
                "lambda_safe": {
                    "distribution": "uniform",
                    "min": 1e-3,
                    "max": 1e-1
                },
                "lambda_unsafe": {
                    "distribution": "uniform",
                    "min": 0.1,
                    "max": 20.0
                },
                "lambda_unsafe_max": {
                    "distribution": "uniform",
                    "min": 0.1,
                    "max": 20.0
                },
                "lambda_cbf": {
                    "distribution": "uniform",
                    "min": 0.1,
                    "max": 200.0
                },
                "lambda_bndry": {
                    "value": 0.0
                },
                "unsafe_margin": {
                    "distribution": "uniform",
                    "min": 0.0,
                    "max": 1.0
                },
                "safe_margin": {
                    "distribution": "uniform",
                    "min": 0.0,
                    "max": 1.0
                },
                "cbf_margin": {
                    "distribution": "uniform",
                    "min": 0.0,
                    "max": 1.0
                },
                "proportions": {
                    "value": {
                        "safe": 0.01,
                        "unsafe": 0.01,
                        "boundary": 0.1
                }
                }
            }
        }
    return sweep_config


def run_sweep_for_system(system_type, num_runs, project_name):
    # Create sweep configuration with continuous distributions
    sweep_config = create_sweep_config(system_type, sweep_method="bayes")

    print("\nSweep configuration:")
    print(f"  Method: {sweep_config['method']}")
    print(f"  Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
    print(f"  Parameters to tune: {len(sweep_config['parameters'])}")

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    print(f"\n✅ Sweep created with ID: {sweep_id}")
    print(f"View sweep at: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}")

    # Run the sweep
    print(f"\nStarting sweep agent for {system_type} (runs: {num_runs})...")
    wandb.agent(sweep_id, function=train_with_config, count=num_runs, project=project_name)

    print("\nSWEEP COMPLETED")
    print(f"View results at: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}")


def main(system_type="barr1", num_runs=50, project_name="cbf-hyperparameter-tuning"):
    """
    Run hyperparameter tuning sweep for CBF training.
    
    IMPORTANT: This sweep optimizes for CBF VALIDITY, not just test accuracy.
    Each trial runs EMPIRICAL validation (50K samples) to count violations and
    assess how well the trained CBF satisfies the barrier function conditions.
    
    Args:
        system_type: Type of dynamical system ("barr1", "barr2", "barr3", "barr4", "hiord2", "hiord4", "hiord6", "hiord8", "simple2d", "cartpole", "rendezvous", or "all" to run all sequentially)
        num_runs: Number of sweep runs to execute
        project_name: W&B project name for the sweep
    """
    
    print("="*60)
    print("CBF HYPERPARAMETER TUNING")
    print("="*60)
    print(f"System: {system_type}")
    print(f"Number of runs: {num_runs}")
    print(f"W&B project: {project_name}")
    print("="*60)
    
    # Support running all systems sequentially
    if system_type == 'all':
        systems = ['hiord6', 'hiord8', 'hiord4'] #, 'simple2d', 'barr2', 'barr3', 'barr4' working, given up on 'barr1' for now
        for st in systems:
            print("\n" + "="*60)
            print(f"Running sweep for {st}")
            print("="*60)
            run_sweep_for_system(st, num_runs, project_name)
        print("\nBest parameter files saved under data/cbf_best_params/")
    else:
        run_sweep_for_system(system_type, num_runs, project_name)


if __name__ == "__main__":
    import argparse
    
    # Set multiprocessing start method for CUDA compatibility
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for CBF training (fossil dynamics)')
    parser.add_argument('--system', type=str, default='rendezvous', 
                       choices=['barr1', 'barr2', 'barr3', 'barr4', 'hiord2', 'hiord4', 'hiord6', 'hiord8', 'simple2d', 'cartpole', 'rendezvous', 'all'],
                       help='Dynamical system type (or "all" to tune all)')
    parser.add_argument('--num-runs', type=int, default=50,
                       help='Number of configurations to try')
    parser.add_argument('--project', type=str, default='cbf-hyperparameter-tuning',
                       help='W&B project name')
    
    args = parser.parse_args()
    
    main(
        system_type=args.system,
        num_runs=args.num_runs,
        project_name=args.project
    )
