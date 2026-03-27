# Scalable Verification of Neural Control Barrier Functions Using Linear Bound Propagation 

This repository contains the code and data underlying the publication "Scalable Verification of Neural Control Barrier Functions Using Linear Bound Propagation" by Nikolaus Vertovec, Frederik Baymler Mathiesen, Thom Badings, Luca Laurenti, and Alessandro Abate. The idea is to provide a framework for verifying that a neural network (also called candidate function) satisfies the control barrier function (CBF) condition for a continuous-time control-affine dynamical system. We say that the neural network candidate is a CBF if it satisfies the condition. 

The code is based on Linear Bound Propagation for input/output-relations and partial derivatives of the neural candidate, and on certified first-order Taylor expansions of the non-linear dynamics.

## Prerequisites
- Docker

## Reproducibility
We provide this respository as a reproducibility package for the publication. We provide .pth and .onnx files for each benchmark in addition to scripts to train the neural network candidate. Then, based on these model files and dynamics models, we verify whether the CBF condition holds.

We provide a Dockerfile to run all benchmarks. To access the Docker image, first build a Docker image with the following command:
```bash
docker build -t ubuntu:lbp-neural-cbf .
```

Then start a container with:
```bash
docker run --name lbp-neural-cbf --rm -v $(pwd)/:/lbp-neural-cbf/ -it ubuntu:lbp-neural-cbf
```

Inside the container, the initial directory is `/lbp-neural-cbf`, which contains the contents of this repository.

#  python3 experiments/barrier_certificate.py

### Training + verification with our method
To run the verification on a given benchmark, execute the following command: `python3 experiments/barrier_certificate.py`

Executing the above will perform the verification with both our proposed approach, using a simplicial mesh, on the default system `"simple2d"`. If you want to train the networks from scratch before verifying them, modify `main` in `barrier_certificate.py` to read `train=True` (can be done in command line via `nano` or directly in the reposity, which is mapped into the container via a mounted volume).

The output is expected to look like the following:
```
============================================================
NEURAL CONTROL BARRIER FUNCTION EXPERIMENT
============================================================
Using Simple 2D System (with constant control: g(x) = I)
System parameters:
  Input dimension: 2
  Control dimension: 2
  Control bounds: u ∈ [[-0.5 -0.5], [0.5 0.5]]
  Safe set: ComplementDomain(dim=2)
  Alpha parameter: 1.0
  Input domain: BoxDomain(dim=2)

----------------------------------------
VERIFICATION PHASE
----------------------------------------
Verifying network: data/simple_2d_cbf.onnx
Verification parameters:
  executor_type: single
  region_type: simplicial

Using simplicial regions and single executor for modular CBF verification...
Verifying CBF: data/simple_2d_cbf.onnx
Using CPU for verification
Using single executor
Overall Progress (remaining samples: 0, certified: 100.0000%, uncertified: 0.0000%): 4608it [00:12, 379.49it/s] 

============================================================
CBF VERIFICATION RESULTS
============================================================
System: simple_2d
Certified percentage: 100.0000%
Uncertified percentage: 0.0000%
Computation time: 12.14 seconds
Total samples processed: 4608
Iterations per second: 379.49 it/s
============================================================

✅ CBF verification PASSED - No counterexamples found!
```

To change the benchmark, modify `main` to use the keyword argument `system_type="<other_system>"`. The options are:
- `"simple2d"` - The Control-2D benchmark
- `"barr1"` - The Darboux benchmark
- `"barr2"` - The Barrier 2 benchmark
- `"barr3"` - The Barrier 3 benchmark
- `"barr4"` - The Barrier 4 benchmark
- `"cartpole"` - The Cart-Pole benchmark

### FOSSIL / dReal
To run FOSSIL (with its dReal-based verifier) for the `"barr1"`, `"barr2"`, `"barr3"`, or `"barr4"` benchmarks, execute the following commands respectively:
- `timeout 1h python3 fossil/experiments/barr_1.py`
- `timeout 1h python3 fossil/experiments/barr_2.py`
- `timeout 1h python3 fossil/experiments/barr_3.py`
- `timeout 1h python3 fossil/experiments/barr_4.py`

The expected output is:
```
unsafe
lie
{'found': True, 'cex': {'unsafe': [], 'lie': []}}
Verifier times: total=489.9562490120006s,min=1.178930711999783s,max=488.77731830000084s,avg=244.9781245060003s, N=2
```
The relevant time is `total`. If the verifier exceeds the alotted time of 1h, the program is terminated without input; typically after `lie` (i.e. while processing the Lie derivative/CBF condition).

When `found == True`, then the candidate satisfies the CBF conditions. 