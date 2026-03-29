# Research Idea Discovery Report - Advanced CSR Extensions

**Direction**: Advanced extensions to Certified-Subspace Repair (CSR) for Neural Control Barrier Functions
**Generated**: 2026-03-29
**Base Method**: CSR uses LBP verification and generalized eigenvalue decomposition for subspace-based repair with verified-region invariance guarantees.

---

## Literature Landscape Summary

### Current State of the Field

**Existing Foundation Works**:
1. **Vertevec et al. (2025)**: Scalable Verification of Neural CBFs using LBP
   - LBP verification produces linear bounds h̲(x) = A_L x + b_L
   - No repair mechanism provided
   - Scales to 6D systems with ~3M simplices

2. **Chen et al. (ACC 2024)**: Verification-Aided Learning with Termination Guarantees
   - CEGIS (Counterexample-Guided Inductive Synthesis)
   - Only repairs last layer via convex optimization
   - Has termination guarantees but limited expressivity

3. **ISAR (2024)**: Repairing Controllers While Preserving What Works
   - Preserves verified regions via simulation-based constraints
   - For general controllers, not CBF-specific
   - No formal invariance guarantees

4. **FaVeR (IJCAI 2025)**: Efficient Counterexample-Guided Fairness Verification and Repair
   - Neuron-level selective repair
   - Based on gradient sensitivity, not certificate structure
   - No verified-region preservation guarantees

5. **SABR (ICLR 2023)**: Certified Training via Small Boxes
   - Small boxes tighten verification bounds during training
   - Not applied to repair phase
   - No subspace decomposition

**Key Gaps Identified**:
1. **No iterative certificate refinement**: Existing methods are one-shot
2. **No adaptive subspace selection**: Fixed subspace dimension (k)
3. **No certificate-gradient alignment**: Repair gradient doesn't consider certificate structure
4. **No multi-certificate repair**: Single CBF, no synergistic effects
5. **No hierarchical subspace**: Flat decomposition, no multi-scale analysis
6. **No gradient invariance exploitation**: The invariance property not fully leveraged

---

## Generated Research Ideas

### 🥇 Idea 1: Iterative Certificate-Gradient Aligned Refinement (ICGAR)

**Summary**: Iteratively refine certificates using gradient-aligned subspace updates that explicitly preserve certificate invariance while tightening failure regions.

**Core Hypothesis**:
Certificate repair can be framed as a manifold-constrained optimization problem where the manifold is defined by verified-region invariance. By aligning repair gradients with the certificate manifold's tangent space, we can achieve faster convergence and better bound tightening.

**Technical Innovation**:

1. **Certificate Manifold Characterization**:
   ```
   Given verified regions V = {v_1, ..., v_m}
   Define certificate manifold M = {θ | h̲_v(θ) = h̲_v(θ_0), ∀v∈V}

   Tangent space T_θ M = span{∂h̲_v/∂θ | v∈V}
   Normal space N_θ M = ker(T_θ M)
   ```

2. **Gradient Projection Repair**:
   ```
   Standard repair: θ_{t+1} = θ_t - η ∇θ L(θ_t)

   Manifold-constrained:
   g = ∇θ L(θ_t)  # repair loss gradient
   g_⊥ = P_N(g)    # project to normal space (violates invariance!)
   g_∥ = P_T(g)    # project to tangent space (preserves invariance!)

   θ_{t+1} = θ_t - η (g_∥ + α g_⊥)
   ```

3. **Adaptive α(t) Schedule**:
   - Early iterations: α ≈ 0 (strict invariance)
   - Later iterations: α > 0 (controlled violation)
   - Based on failure region reduction rate

**Why This Is Novel**:
1. First to formulate certificate invariance as an explicit manifold constraint
2. Gradient-aligned repair provides faster convergence than random subspace
3. Adaptive α allows controlled trade-off between repair and preservation
4. Tangent space projection is provably optimal for manifold-constrained optimization

**Minimum Viable Experiment**:

```python
# Setup
network = load_cbf_model('double_integrator')
verified, failed = verify_with_lbp(network)

# Compute certificate manifold
tangent_space = compute_certificate_tangent(verified, network)

# Iterative repair
for iter in range(100):
    # Compute repair loss (on failed regions)
    loss = repair_loss(network, failed)
    grad = torch.autograd.grad(loss, network.parameters())

    # Project to tangent space
    grad_tangent = project_to_tangent(grad, tangent_space)

    # Update with adaptive alpha
    alpha = compute_alpha(iter, failure_reduction_rate)
    grad_adjusted = grad_tangent + alpha * (grad - grad_tangent)

    # Update network
    for p, g in zip(network.parameters(), grad_adjusted):
        p.data -= lr * g

    # Re-verify every 10 iterations
    if iter % 10 == 0:
        verified_new, failed_new = verify_with_lbp(network)
        if len(failed_new) == 0:
            break

# Verify 100% invariance
assert verified_regions_preserved(verified, verified_new)
```

**Metrics to Track**:
- Repair convergence rate (iterations to fix failures)
- Certificate invariance violation (max |h̲_v_new - h̲_v_old|)
- Final verified region ratio
- Bound gap reduction in failure regions

**Expected Contribution Type**: New method with theoretical grounding

**Risk Level**: MEDIUM
- **Why**: Manifold projection is well-understood, but applying to CBF certificates is novel
- **Mitigation**: Start with 2D systems, verify projection correctness

**Estimated Effort**: 4-6 weeks
- Week 1-2: Manifold characterization and projection implementation
- Week 3-4: Adaptive α tuning and baseline experiments
- Week 5-6: Full evaluation on 2D/4D/6D benchmarks

**Feasibility**: HIGH
- Uses existing verification code
- Compute: <1 GPU day for all experiments
- Data: Standard CBF benchmarks (Double Integrator, Bicycle, Quadrotor)

**Reviewer's Likely Objection**:
"How do you compute the tangent space efficiently for large numbers of verified regions?"

**Response**:
We use a key insight: the tangent space is low-dimensional. For most CBFs, the certificate manifold has dimension << #parameters. We compute it via SVD of the Jacobian J = [∂h̲_v/∂θ] for all v∈V, keeping only top-k singular vectors. This reduces complexity from O(m·d·|θ|) to O(k·|θ|).

**Why We Should Do This**:
This provides a principled, theoretically-grounded way to perform certificate repair. The manifold formulation gives us a clean way to think about the invariance property, and gradient projection gives us a practical algorithm. This could be the theoretical foundation for the entire CSR family.

---

### 🥈 Idea 2: Adaptive Subspace Selection via Failure-Mode Clustering (ASSC)

**Summary**: Automatically discover multiple failure modes via spectral clustering, then perform targeted subspace repair for each mode separately.

**Core Hypothesis**:
Failed regions often have distinct underlying causes (e.g., high-velocity failures vs. position errors). By clustering failures based on their A_L spectral signatures, we can discover these modes and apply mode-specific repairs that don't interfere with each other.

**Technical Innovation**:

1. **Spectral Signature Extraction**:
   ```
   For each failed region f, compute its signature:
   σ_f = eigendecomposition(A_Lf^T A_Lf)

   Signature space S = {σ_f | f∈F}
   ```

2. **Spectral Clustering**:
   ```
   Cluster S into C = {C_1, ..., C_k} where:
   - Within-cluster signatures are similar (low spectral distance)
   - Between-cluster signatures are different (high spectral distance)

   Use: normalized spectral clustering with automatic k selection
   ```

3. **Mode-Specific Subspace Decomposition**:
   ```
   For each cluster C_i:
       V_verified = all verified regions
       F_i = {f | σ_f ∈ C_i}

       Compute subspace decomposition:
       M_V = 1/|V| Σ A_Lv^T A_Lv
       M_Fi = 1/|F_i| Σ A_Lf^T A_Lf

       Solve: M_Fi w = λ M_V w
       Take top-k eigenvectors → subspace S_i
   ```

4. **Staged Repair**:
   ```
   For each cluster C_i in order of failure severity:
       1. Repair only in subspace S_i
       2. Verify results
       3. Update verified regions V_verified
       4. Repeat for next cluster
   ```

**Why This Is Novel**:
1. First to analyze failure modes via spectral clustering of A_L signatures
2. Mode-specific repairs are more targeted and less likely to interfere
3. Automatic cluster count selection avoids manual tuning
4. Staged repair guarantees monotonic improvement

**Minimum Viable Experiment**:

```python
# Setup
network = load_cbf_model('bicycle_4d')
verified, failed = verify_with_lbp(network)

# Step 1: Extract spectral signatures
signatures = []
for region in failed:
    A_L = get_A_L_matrix(region)
    eigvals = eigvals(A_L.T @ A_L)
    signatures.append(eigvals)

# Step 2: Cluster signatures
from sklearn.cluster import SpectralClustering

# Auto-select k via eigengap
clusters = spectral_clustering_auto_k(signatures)

# Step 3: Mode-specific repair
repaired_network = network
for cluster_id in range(max(clusters) + 1):
    cluster_failures = [f for f, c in zip(failed, clusters) if c == cluster_id]

    if not cluster_failures:
        continue

    # Compute cluster-specific subspace
    subspace = compute_cluster_subspace(verified, cluster_failures)

    # Repair in this subspace
    repaired_network = repair_in_subspace(
        repaired_network,
        cluster_failures,
        subspace
    )

    # Update verified regions
    verified, failed = verify_with_lbp(repaired_network)

    print(f"Cluster {cluster_id}: {len(cluster_failures)} → {len(failed)} failures remaining")

# Final verification
verified_final, failed_final = verify_with_lbp(repaired_network)
print(f"Final: {len(failed_final)} failures")
```

**Metrics to Track**:
- Number of discovered failure modes (k)
- Cluster separation quality (silhouette score)
- Per-cluster repair effectiveness
- Total repair time vs. baseline CSR

**Expected Contribution Type**: Diagnostic + method extension

**Risk Level**: LOW
- **Why**: Clustering is well-understood; we're just applying it to a new domain
- **Mitigation**: Validate clustering on synthetic cases first

**Estimated Effort**: 3-4 weeks
- Week 1: Signature extraction and clustering implementation
- Week 2: Cluster-specific subspace decomposition
- Week 3: Staged repair pipeline
- Week 4: Evaluation and comparison

**Feasibility**: HIGH
- Uses standard ML libraries (scikit-learn)
- Compute: <2 GPU hours for all experiments
- No new data needed

**Reviewer's Likely Objection**:
"How do you know that spectral clustering is the right approach? What if failures aren't spectrally separable?"

**Response**:
We provide an ablation study comparing:
1. Spectral clustering on A_L signatures
2. K-means clustering on A_L signatures
3. No clustering (baseline CSR)

We also study cases where clustering fails (single dominant mode) and show that the method gracefully degrades to baseline behavior. The spectral approach is motivated by the fact that A_L signatures capture the "certificate gradient direction," which is a principled way to characterize failure modes.

**Why We Should Do This**:
This idea takes CSR from "one-size-fits-all" to "precision medicine." By understanding why failures occur and treating them differently, we can achieve better repair with less effort. It also provides diagnostic value: the cluster assignments tell us what kinds of failures the CBF has.

---

### 🥉 Idea 3: Hierarchical Multi-Scale Subspace Decomposition (HMSSD)

**Summary**: Decompose the weight space hierarchically at multiple scales, allowing coarse-to-fine repair that first fixes large-scale certificate violations and then fine-tunes local behavior.

**Core Hypothesis**:
Certificate violations occur at different scales: some are global (network is wrong everywhere), some are regional (specific area of state space), and some are local (small neighborhoods). A hierarchical subspace decomposition allows us to fix violations in order of scale, ensuring that coarse fixes don't inadvertently break fine structure.

**Technical Innovation**:

1. **Wavelet-Based Subspace Decomposition**:
   ```
   Instead of standard eigenvalue decomposition, use wavelet transform:

   For each weight matrix W_l in layer l:
       ψ_l = wavelet_transform(W_l)

   Organize by scale:
       W_l = Σ_s Φ_s(ψ_l[s])  # s = scale
   Where Φ_s projects to scale-s subspace
   ```

2. **Scale-Aware Failure Analysis**:
   ```
   For each failed region f with A_Lf:
       Compute "scale signature" s_f:
       s_f = argmax_s |Φ_s(∂h̲_f/∂θ)|

       This tells us which scale dominates the violation
   ```

3. **Coarse-to-Fine Repair**:
   ```
   For scale s in [coarse, ..., fine]:
       1. Select failures F_s dominated by scale s
       2. Repair only in scale-s subspace
       3. Verify
       4. Freeze scale-s subspace
       5. Move to next scale
   ```

4. **Theoretical Guarantee**:
   ```
   Proposition: If we repair at scale s in subspace Φ_s,
   then all finer scales s' < s are invariant up to wavelet precision.

   Corollary: Coarse repairs preserve fine-grained certificate structure.
   ```

**Why This Is Novel**:
1. First to apply wavelet/multi-scale analysis to certificate repair
2. Coarse-to-fine repair is a natural progression from the repair problem structure
3. Wavelet decomposition provides explicit scale control
4. Theoretical invariance guarantees across scales

**Minimum Viable Experiment**:

```python
import pywt

# Setup
network = load_cbf_model('double_integrator')
verified, failed = verify_with_lbp(network)

# Step 1: Wavelet decomposition of weights
wavelet_network = {}
for name, param in network.named_parameters():
    if 'weight' in name:
        # 2D wavelet transform
        coeffs = pywt.wavedec2(param.data.numpy(), 'db4', level=3)
        wavelet_network[name] = coeffs

# Step 2: Analyze failure scales
failure_scales = []
for region in failed:
    # Compute gradient w.r.t weights
    grad = compute_certificate_gradient(network, region)

    # Determine dominant scale
    max_norm = 0
    dominant_scale = None
    for scale_idx, scale_coeff in enumerate(grad_wavelet_coeffs):
        norm = torch.norm(scale_coeff)
        if norm > max_norm:
            max_norm = norm
            dominant_scale = scale_idx

    failure_scales.append(dominant_scale)

# Step 3: Coarse-to-fine repair
repaired_network = network
frozen_scales = set()

for scale in range(4):  # 3 wavelet levels + 1 residual
    # Select failures at this scale
    scale_failures = [f for f, s in zip(failed, failure_scales) if s == scale]

    if not scale_failures:
        frozen_scales.add(scale)
        continue

    # Repair only at this scale
    mask = create_scale_mask(wavelet_network, scale, frozen_scales)
    repaired_network = repair_with_mask(
        repaired_network,
        scale_failures,
        mask
    )

    # Freeze this scale
    frozen_scales.add(scale)

    # Verify
    verified, failed = verify_with_lbp(repaired_network)
    print(f"Scale {scale}: {len(scale_failures)} → {len(failed)} remaining")

# Final verification
verified_final, failed_final = verify_with_lbp(repaired_network)
```

**Metrics to Track**:
- Scale distribution of failures (what fraction at each scale?)
- Per-scale repair effectiveness
- Total repair iterations vs. baseline
- Boundary gap reduction per scale

**Expected Contribution Type**: New method with multi-scale analysis

**Risk Level**: MEDIUM
- **Why**: Wavelets are standard but applying to neural weights for repair is novel
- **Mitigation**: Start with fixed wavelet basis, validate on 2D

**Estimated Effort**: 5-6 weeks
- Week 1-2: Wavelet decomposition and mask creation
- Week 3-4: Scale-aware failure analysis
- Week 5-6: Coarse-to-fine repair and evaluation

**Feasibility**: MEDIUM
- Requires implementing wavelet operations on PyTorch tensors
- Compute: <1 GPU day
- Complexity: Managing wavelet coefficient updates correctly

**Reviewer's Likely Objection**:
"Why wavelets specifically? Why not PCA or some other multi-scale method?"

**Response**:
We provide an ablation comparing:
1. Wavelet-based decomposition
2. SVD-based multi-scale (top-k components)
3. Flat decomposition (baseline CSR)

Wavelets have advantages: they provide explicit scale interpretation, are computationally efficient (O(n)), and have good localization properties. We report which method works best, but we hypothesize wavelets due to their scale-localization trade-off.

**Why We Should Do This**:
This idea provides a principled way to think about "what scale" of repair is needed. If failures are coarse, we should fix coarsely; if they're fine, we should fix finely. The hierarchical approach gives us this flexibility and provides guarantees that coarse fixes don't destroy fine structure.

---

### Idea 4: Certificate-Gradient Invariance Exploration (CGIE)

**Summary**: Empirically study the invariance property of certificate gradients across state space and parameter space, leading to theoretical insights that could improve all repair methods.

**Core Hypothesis**:
Certificate gradients ∂h̲(x)/∂θ have special structure: they vary slowly in state space but span a low-dimensional subspace in parameter space. Characterizing this structure quantitatively could lead to stronger theoretical guarantees for CSR and related methods.

**Technical Innovation**:

1. **Gradient State-Space Smoothness**:
   ```
   For neighboring simplices v, v' (adjacent in triangulation):
       Δ_state = v.center - v'.center
       Δ_grad = ∂h̲_v/∂θ - ∂h̲_v'/∂θ

       Analyze: ||Δ_grad|| / ||Δ_state||
       Hypothesis: This ratio is small → smooth
   ```

2. **Gradient Parameter-Space Structure**:
   ```
   For all verified simplices {v_i}:
       G = [∂h̲_v_i/∂θ]  # Stack as rows

   Compute SVD: G = U Σ V^T

   Analyze:
       - Rank(G) vs. |θ| (effective dimensionality)
       - Decay of singular values (low-rank structure?)
       - Relationship to A_L signature
   ```

3. **Invariance Mechanisms**:
   ```
   Compare three invariance definitions:
       1. CSR invariance: h̲_v(θ') = h̲_v(θ)
       2. Gradient invariance: ∂h̲_v/∂θ = ∂h̲_v/∂θ₀
       3. Value invariance: h̲_v(x; θ') = h̲_v(x; θ₀)

   Study: Which is achievable? Which is sufficient?
   ```

4. **Theoretical Implications**:
   ```
   If gradients are low-rank:
       → Certificate function is "simple" in parameter space
       → Repair problem is tractable
       → CSR subspace selection is principled

   If gradients are state-smooth:
       → Failures are "localized" in state space
       → Local repair should work well
   ```

**Why This Is Novel**:
1. First systematic study of certificate gradient structure
2. Connects LBP verification gradients to parameter space geometry
3. Could provide theoretical foundation for multiple repair methods
4. Empirical findings are valuable regardless of theoretical outcome

**Minimum Viable Experiment**:

```python
# Setup
networks = [
    load_cbf_model('double_integrator'),
    load_cbf_model('bicycle_4d'),
    load_cbf_model('quadrotor_6d')
]

for network in networks:
    verified, failed = verify_with_lbp(network)

    # Study 1: State-space smoothness
    smoothness_ratios = []
    for v1, v2 in get_adjacent_simplices(verified):
        grad1 = compute_certificate_gradient(network, v1)
        grad2 = compute_certificate_gradient(network, v2)

        state_dist = torch.norm(v1.center - v2.center)
        grad_dist = torch.norm(grad1 - grad2)

        smoothness_ratios.append(grad_dist / state_dist)

    print(f"State smoothness: mean={np.mean(smoothness_ratios):.4f}, std={np.std(smoothness_ratios):.4f}")

    # Study 2: Parameter-space structure
    grad_matrix = []
    for v in verified:
        grad = compute_certificate_gradient(network, v)
        grad_matrix.append(grad.flatten().numpy())

    grad_matrix = np.array(grad_matrix)  # shape: (m, |θ|)

    U, s, Vt = np.linalg.svd(grad_matrix, full_matrices=False)

    print(f"Parameter structure: rank={np.linalg.matrix_rank(grad_matrix)}, param_dim={grad_matrix.shape[1]}")
    print(f"Singular value decay: {s[:10]}")

    # Study 3: Invariance mechanisms
    test_invariance_mechanisms(network, verified)

def test_invariance_mechanisms(network, verified):
    # Perturb in CSR subspace
    delta_csr = generate_csr_perturbation(network, verified)

    # Compare invariance measures
    csr_value_inv = compute_value_invariance(network, delta_csr, verified)
    csr_grad_inv = compute_grad_invariance(network, delta_csr, verified)

    print(f"CSR value invariance: {csr_value_inv:.6f}")
    print(f"CSR grad invariance: {csr_grad_inv:.6f}")
```

**Metrics to Track**:
- State-space gradient smoothness statistics
- Parameter-space gradient rank and singular value decay
- Correlation between A_L signatures and gradient signatures
- Invariance metric comparison

**Expected Contribution Type**: Diagnostic + theoretical foundation

**Risk Level**: LOW
- **Why**: Pure empirical study with clear deliverables
- **Mitigation**: If structure doesn't exist, that's a valid negative result

**Estimated Effort**: 3-4 weeks
- Week 1: Gradient computation infrastructure
- Week 2: State-space analysis
- Week 3: Parameter-space analysis
- Week 4: Invariance comparison and synthesis

**Feasibility**: HIGH
- Uses existing verification code
- Compute: <1 GPU hour
- No new methods required

**Reviewer's Likely Objection**:
"This seems like an explor study. What's the main contribution?"

**Response**:
The main contribution is a characterization that could enable multiple future works. Specifically:
1. If we find strong low-rank structure, we can propose more efficient algorithms
2. If we find state-space smoothness, we can propose local repair methods
3. The invariance comparison tells us which guarantees are achievable

We frame the paper as "Understanding Certificate Repair: A Gradient-Structure Perspective," with a theoretical section deriving implications from our empirical findings.

**Why We Should Do This**:
This idea provides the fundamental understanding that underpins all repair methods. Before we build more complex algorithms (like Ideas 1-3), we should understand the problem structure. The insights could lead to a theoretical framework for certificate repair that multiple papers could build on.

---

### Idea 5: Multi-Certificate Synergistic Repair (MCSR)

**Summary**: Train and repair multiple CBFs simultaneously, exploiting synergies between certificates to achieve better overall safety guarantees.

**Core Hypothesis**:
Different CBFs encode different aspects of safety (e.g., distance to obstacle vs. velocity bounds). By training and repairing them jointly, we can exploit synergies: what one certificate learns, others can use, reducing total training/repair cost.

**Technical Innovation**:

1. **Joint Certificate Framework**:
   ```
   Define K certificates: h¹(x), ..., h^K(x)

   Joint CBF condition:
       α_k(h_k(x)) ≥ sup_u[L_f h_k(x, u)], ∀k=1..K

   Synergy mechanism:
       Each certificate h_k has "privileged features" F_k
       Share these features between certificates
   ```

2. **Synergistic Subspace Decomposition**:
   ```
   For each certificate k:
       Compute its failure subspace F_k
       Compute its verified subspace V_k

   Joint repair space:
       F_joint = span(F_1, ..., F_K)  # Union of all failure subspaces
       V_joint = ∩ V_k             # Intersection of verified subspaces

   Synergy condition:
       F_joint should be "small" (few shared failure modes)
       V_joint should be "large" (many shared verified modes)
   ```

3. **Alternating Repair Schedule**:
   ```
   Repeat until all certificates verified:
       1. Select worst-failing certificate k*
       2. Repair h_k* in F_joint subspace
       3. Update all certificates using shared features
       4. Re-verify all certificates
   ```

**Why This Is Novel**:
1. First to consider multiple CBFs in repair
2. Synergy mechanism is principled (feature sharing)
3. Joint subspace provides new invariance guarantees
4. Alternating schedule is novel in CBF context

**Minimum Viable Experiment**:

```python
# Setup: Multiple certificates for same system
certificates = {
    'distance': load_cbf_model('double_integrator_distance'),
    'velocity': load_cbf_model('double_integrator_velocity'),
    'combined': load_cbf_model('double_integrator_combined')
}

# Step 1: Verify each certificate
results = {}
for name, cert in certificates.items():
    verified, failed = verify_with_lbp(cert)
    results[name] = {'verified': verified, 'failed': failed}

# Step 2: Analyze synergy
for name1, cert1 in certificates.items():
    for name2, cert2 in certificates.items():
        if name1 >= name2:
            continue

        # Compare certificate subspaces
        subspace_1 = get_certificate_subspace(cert1, results[name1])
        subspace_2 = get_certificate_subspace(cert2, results[name2])

        # Compute synergy metrics
        shared_fail = compute_subspace_overlap(
            subspace_1['failure'],
            subspace_2['failure']
        )
        shared_verified = compute_subspace_overlap(
            subspace_1['verified'],
            subspace_2['verified']
        )

        print(f"{name1} vs {name2}: shared_fail={shared_fail:.2f}, shared_verified={shared_verified:.2f}")

# Step 3: Joint repair (if synergy exists)
if has_synergy(results):
    # Create joint network
    joint_network = create_multi_certificate_network(certificates)

    # Alternating repair
    for iteration in range(50):
        # Find worst certificate
        worst_cert = find_worst_certificate(joint_network)

        # Repair in joint subspace
        joint_subspace = compute_joint_repair_space(joint_network)
        joint_network = repair_in_subspace(
            joint_network,
            worst_cert,
            joint_subspace
        )

        # Verify all
        all_verified = verify_all_certificates(joint_network)
        if all_verified:
            break
```

**Metrics to Track**:
- Inter-certificate subspace overlap
- Joint repair vs. independent repair effectiveness
- Training time with synergy vs. without
- Safety guarantee improvement

**Expected Contribution Type**: New method with theoretical analysis

**Risk Level**: HIGH
- **Why**: Multi-certificate setup is complex; synergy might not exist
- **Mitigation**: Start with 2 certificates where synergy is expected

**Estimated Effort**: 6-8 weeks
- Week 1-2: Multi-certificate infrastructure
- Week 3-4: Synergy analysis and formulation
- Week 5-6: Joint repair algorithm
- Week 7-8: Evaluation

**Feasibility**: MEDIUM
- Requires training multiple CBFs
- Compute: 2-3 GPU days
- Data: Need to define meaningful certificate tasks

**Reviewer's Likely Objection**:
"Why would we want multiple certificates? Isn't one sufficient?"

**Response**:
Multiple certificates are useful when:
1. Safety has multiple aspects (e.g., obstacle avoidance + velocity limits)
2. Different certificates use different state-space partitionings
3. We want to verify multiple properties independently

We demonstrate on systems where multiple certificates are natural (e.g., robot arm with joint limits, velocity limits, and obstacle constraints). If synergy exists, we show significant efficiency gains. If not, we report that as a negative result.

**Why We Should Do This**:
This is a "high risk, high reward" idea. If certificate synergies exist, this could enable new applications where multiple safety properties need to be verified simultaneously. Even if synergies don't exist, understanding why not is valuable.

---

## Ranked Ideas

### 🥇 **Idea 1: Iterative Certificate-Gradient Aligned Refinement (ICGAR)** — **RECOMMENDED**

**Rationale**:
- **Theoretical grounding**: Explicit manifold formulation gives clean theoretical foundation
- **Practical innovation**: Gradient projection provides efficient algorithm
- **Novelty**: First to frame certificate invariance as manifold constraint
- **Impact**: Could become theoretical foundation for CSR family
- **Risk**: MEDIUM (mitigated by starting with 2D)
- **Effort**: 4-6 weeks (reasonable)

**Key Innovation**: Gradient-aligned repair on certificate manifold
**Expected Contribution**: New method + theoretical analysis
**Pilot Complexity**: Low (2D system, <1 GPU hour)

---

### 🥈 **Idea 2: Adaptive Subspace Selection via Failure-Mode Clustering (ASSC)** — **BACKUP**

**Rationale**:
- **Diagnostic value**: Clustering reveals failure mode structure
- **Practical benefit**: Mode-specific repairs are more targeted
- **Novelty**: First to analyze failure modes via spectral clustering
- **Impact**: Takes CSR from "one-size-fits-all" to precision
- **Risk**: LOW (clustering is well-understood)
- **Effort**: 3-4 weeks (reasonable)

**Key Innovation**: Spectral clustering of A_L signatures
**Expected Contribution**: Diagnostic + method extension
**Pilot Complexity**: Very low (<30 min GPU time)

---

### 🥉 **Idea 3: Hierarchical Multi-Scale Subspace Decomposition (HMSSD)** — **INTERESTING BUT RISKIER**

**Rationale**:
- **Novel perspective**: Multi-scale analysis of repair problem
- **Theoretical guarantee**: Cross-scale invariance property
- **Impact**: Coarse-to-fine repair is intuitive
- **Risk**: MEDIUM (wavelet implementation complexity)
- **Effort**: 5-6 weeks

**Key Innovation**: Wavelet-based multi-scale decomposition
**Expected Contribution**: New method
**Pilot Complexity**: Medium (wavelet setup needed)

---

### 📊 **Idea 4: Certificate-Gradient Invariance Exploration (CGIE)** — **FOUNDATIONAL WORK**

**Rationale**:
- **High impact**: Could enable multiple future works
- **Low risk**: Pure empirical study, deliverables guaranteed
- **Novelty**: First systematic study of certificate gradient structure
- **Impact**: Provides theoretical foundation
- **Risk**: LOW (negative results are valid)
- **Effort**: 3-4 weeks (fast)

**Key Innovation**: Gradient-structure characterization
**Expected Contribution**: Diagnostic + theoretical foundation
**Pilot Complexity**: Very low (<1 GPU hour)

---

## Eliminated Ideas

| Idea | Reason Eliminated |
|------|-------------------|
| Multi-Certificate Synergistic Repair | High complexity, unclear if synergies exist; not enough benefit for risk |
| Simple subspace extension | Too incremental over base CSR; needs more novelty |

---

## Comparison to Base CSR

| Dimension | Base CSR | ICGAR (Idea 1) | ASSC (Idea 2) | HMSSD (Idea 3) |
|-----------|-----------|-------------------|------------------|------------------|
| Theoretical Basis | Eigenvalue decomposition | Manifold optimization | Clustering | Wavelet theory |
| Repair Strategy | One-shot fixed subspace | Iterative gradient-aligned | Mode-specific staged | Coarse-to-fine |
| Invariance Guarantee | 100% verified regions | Tangent space projection | Per-cluster invariance | Cross-scale invariance |
| Novelty | First method | Enhanced theoretical | Failure mode discovery | Multi-scale repair |
| Risk Level | LOW | MEDIUM | LOW | MEDIUM |
| Effort | Baseline | +2-3 weeks | +1-2 weeks | +3-4 weeks |

---

## Suggested Execution Order

### Phase 1: Idea 4 (CGIE) — Foundation
**Why first?**
- Low risk, high impact
- Provides understanding for all other ideas
- Fast results (<2 weeks)

**Timeline**: 3-4 weeks
- Week 1-2: Gradient computation and analysis
- Week 3: Invariance comparison
- Week 4: Paper writing

**Deliverable**: "Understanding Certificate Repair: A Gradient-Structure Perspective"

---

### Phase 2: Idea 1 (ICGAR) — Main Contribution
**Why second?**
- Builds on foundation from Idea 4
- Highest theoretical impact
- Reasonable effort

**Timeline**: 4-6 weeks (after Idea 4)

**Deliverable**: "Iterative Certificate-Gradient Aligned Refinement"

---

### Phase 3: Idea 2 (ASSC) — Extension
**Why third?**
- Low effort, nice addition
- Could be combined with Idea 1
- Provides diagnostic value

**Timeline**: 3-4 weeks (can overlap with Idea 2)

**Deliverable**: Extension paper or section in main paper

---

## Success Criteria

### For Idea 4 (CGIE):
- [ ] Characterize gradient state-space smoothness (quantitative metrics)
- [ ] Characterize gradient parameter-space structure (rank, singular value decay)
- [ ] Compare invariance mechanisms (which are achievable?)
- [ ] Derive at least 2 theoretical implications

### For Idea 1 (ICGAR):
- [ ] Prove manifold invariance theorem
- [ ] Implement gradient projection algorithm
- [ ] Demonstrate faster convergence than baseline CSR
- [ ] Achieve 100% verified-region preservation
- [ ] Show bound tightening benefits

### For Idea 2 (ASSC):
- [ ] Demonstrate distinct failure modes in benchmarks
- [ ] Show mode-specific repair beats baseline
- [ ] Achieve computational efficiency gains
- [ ] Provide diagnostic visualizations

---

## Next Steps

1. **Implement gradient computation** (Week 1 of Idea 4):
   - Hook into LBP verification code
   - Compute ∂h̲(x)/∂θ for each simplex
   - Verify correctness on small example

2. **Run foundational study** (Weeks 2-4 of Idea 4):
   - Analyze gradient structure on all benchmarks
   - Document findings
   - Write draft paper

3. **Based on findings, select path**:
   - If structure is simple: Proceed with Idea 1 (ICGAR)
   - If structure is complex: Consider Idea 2 (ASSC) first

4. **Implement main contribution** (Weeks 1-6 of selected idea):
   - Develop full algorithm
   - Run comprehensive experiments
   - Write full paper

---

## Sources

**Local Papers Analyzed**:
1. Vertovec et al. (2025) - Scalable Verification of Neural CBFs
2. Chen et al. (2024) - Verification-Aided Learning
3. SABR (ICLR 2023) - Certified Training
4. FaVeR (IJCAI 2025) - Fairness Verification and Repair
5. ISAR (2024) - Controller Repair

**Related Web Sources**:
- Literature on neural network verification and repair
- Certificate function methods
- Subspace decomposition techniques
- Gradient-based optimization methods

---

*Generated by Idea Discovery Pipeline: March 29, 2026*
