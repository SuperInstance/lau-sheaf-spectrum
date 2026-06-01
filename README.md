# lau-sheaf-spectrum

**Spectral sheaf theory in Rust: sheaf Laplacians, Hodge decomposition, sheaf diffusion, connection Laplacians, sheaf neural networks, synchronization, and persistent sheaf cohomology.**

This crate implements the intersection of sheaf cohomology and spectral graph theory — a powerful mathematical framework where every vertex in a graph gets a vector space (a "stalk"), edges carry linear maps ("restriction maps"), and the resulting sheaf Laplacian reveals the global structure of the system.

---

## What This Does

Given a graph and a cellular sheaf (vector spaces on vertices + linear maps on edges), this crate computes:

1. **Sheaf Laplacian** — L = DᵀD where D is the coboundary operator. For constant sheaves, this recovers the standard graph Laplacian.
2. **Hodge decomposition** — orthogonal splitting of sections into harmonic (ker L) and image (im Dᵀ) components.
3. **Sheaf diffusion** — heat equation dx/dt = −Lx that converges to the harmonic projection.
4. **Connection Laplacian** — for vector bundles with rotation/orthogonal transition maps on edges.
5. **Sheaf neural networks** — diffusion-based graph learning layers: X′ = σ((I − tL) X W).
6. **Synchronization analysis** — spectral gap, consensus, and synchronizability checking.
7. **Persistent sheaf cohomology** — filtration-based barcode computation for H⁰.

Everything is pure Rust, uses `nalgebra` for linear algebra, and is fully serializable with `serde`.

---

## Key Idea

> **A sheaf on a graph is a local-to-global data structure.** Each vertex stores data in its stalk, restriction maps enforce consistency across edges, and the sheaf Laplacian measures how "harmonious" the global assignment is. Zero eigenvalues = perfectly consistent sections; nonzero eigenvalues = inconsistency that diffusion will smooth out.

---

## Install

```toml
[dependencies]
lau-sheaf-spectrum = "0.1"
```

Requires Rust 2021 edition. Dependencies: `nalgebra` (with serde), `serde`/`serde_json`.

---

## Quick Start

```rust
use lau_sheaf_spectrum::prelude::*;

fn main() {
    // Build a graph
    let graph = Graph::complete(5);

    // Constant sheaf: every vertex gets R², all restriction maps = identity
    let sheaf = Sheaf::constant(&graph, 2);

    // Build the sheaf Laplacian
    let laplacian = SheafLaplacian::build(&sheaf, &graph);
    println!("Kernel dim (harmonic sections): {}", laplacian.kernel_dim());
    println!("Spectral gap: {:.4}", laplacian.spectral_gap().unwrap());

    // Hodge decomposition
    let hodge = HodgeDecomposition::compute(&sheaf, &graph);
    println!("Betti-0: {}", hodge.betti_0());

    // Sheaf diffusion: converge to harmonic section
    let diffusion = SheafDiffusion::new(&sheaf, &graph);
    let x0 = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let xt = diffusion.evolve(&x0, 50.0);
    println!("After diffusion: {:?}", xt.as_slice());

    // Persistent sheaf cohomology
    let ps = PersistentSheaf::new(graph, sheaf);
    let barcode = ps.compute_persistent_h0();
    println!("H⁰ barcode: {} bars", barcode.len());
}
```

---

## API Reference

### `Graph` — undirected weighted graph

| method | description |
|---|---|
| `new(n)` | n isolated vertices |
| `complete(n)` / `path(n)` / `cycle(n)` | factory constructors |
| `add_edge(u, v)` / `add_weighted_edge(u, v, w)` | mutation |
| `laplacian()` | standard graph Laplacian D − A |
| `incidence_matrix()` | oriented incidence matrix |
| `is_connected()` | BFS connectivity check |

### `Sheaf` — cellular sheaf on a graph

| method | description |
|---|---|
| `constant(graph, d)` | every stalk = ℝᵈ, restriction maps = identity |
| `from_raw(dims, edge_dims, maps_data)` | custom restriction maps from flat data |
| `coboundary(graph)` | coboundary operator D (total_edge_dim × total_vertex_dim) |
| `total_vertex_dim()` / `total_edge_dim()` | summed stalk dimensions |

`VectorSheaf`: a sheaf with a concrete section (values at vertices).

### `SheafLaplacian` — L = DᵀD with eigendecomposition

| method | description |
|---|---|
| `build(sheaf, graph)` | construct + decompose |
| `smallest_eigenvalue()` | λ_min (should be ≥ 0 for PSD) |
| `spectral_gap()` | first nonzero eigenvalue |
| `kernel_dim()` | dim ker L = dim H⁰ |
| `fiedler_value()` | algebraic connectivity |

### `HodgeDecomposition` — harmonic + image splitting

| method | description |
|---|---|
| `compute(sheaf, graph)` | build decomposition |
| `project_harmonic(x)` / `project_image(x)` | project onto subspaces |
| `decompose(x)` | (harmonic, image) pair |
| `betti_0()` | dim H⁰ |
| `is_harmonic(x)` | check if section is in ker L |

### `SheafDiffusion` — heat equation dx/dt = −Lx

| method | description |
|---|---|
| `evolve(x0, t)` | exact: x(t) = exp(−tL) x₀ |
| `step(x0, dt, steps)` | discrete Euler: x_{k+1} = (I − dt·L) x_k |
| `converge_to_harmonic(x0, dt, steps, tol)` | iterate + check residual |
| `energy_trajectory(x0, dt, steps)` | E(x) = xᵀLx at each step |
| `convergence_rate()` | spectral gap |

### `ConnectionLaplacian` — vector bundle connections

| method | description |
|---|---|
| `build(graph, d, connections)` | custom transition maps per edge |
| `trivial(graph, d)` | identity connections |
| `spectral_gap()` / `kernel_dim()` | spectral properties |

### `SheafNNLayer` / `SheafConvNet` — sheaf neural networks

Layer: `X′ = σ((I − tL) X W)`

| method | description |
|---|---|
| `SheafNNLayer::new(d_in, d_out, t, activation)` | Xavier-initialized weights |
| `forward(x, laplacian)` | single-layer forward pass |
| `SheafConvNet::new(&[sizes], t, activation)` | multi-layer constructor |
| `forward(x, laplacian)` | multi-layer forward pass |

Activations: `Identity`, `ReLU`, `LeakyReLU(α)`, `Tanh`, `Sigmoid`.

### `ConsensusProblem` — multi-agent consensus

| method | description |
|---|---|
| `standard(graph, d)` | constant-sheaf consensus |
| `solve(x0)` | harmonic projection = consensus limit |
| `disagreement(x)` | ‖x − x̄‖² = xᵀLx |

### `PersistentSheaf` — persistent sheaf cohomology

| method | description |
|---|---|
| `compute_persistent_h0()` | fast union-find barcode |
| `compute_persistent_h0_exact()` | full Laplacian at each step |
| `compute_h1_bars()` | approximate H¹ via Euler characteristic |

`Barcode`: collection of `Bar { birth, death, dim }`. Methods: `len`, `betti_at(t)`.

---

## How It Works

### Architecture

```
Graph + Sheaf
  ├── SheafLaplacian::build()
  │     └── coboundary D → L = DᵀD → eigen-decompose
  ├── HodgeDecomposition
  │     └── ker(L) ⊕ im(Dᵀ) orthogonal split
  ├── SheafDiffusion
  │     └── exp(−tL) via eigendecomposition
  ├── ConnectionLaplacian
  │     └── rotation/orthogonal transition maps
  ├── SheafNNLayer
  │     └── σ((I − tL) X W)
  ├── ConsensusProblem
  │     └── harmonic projection
  └── PersistentSheaf
        └── edge-weight filtration → barcode
```

### Coboundary Operator

For edge e = (u, v) with restriction maps F_{u→e} and F_{v→e}:

Dₑ = [−F_{u→e} | F_{v→e}]

The full coboundary D stacks these block-rows for all edges. The sheaf Laplacian L = DᵀD is always positive semidefinite.

### Eigen-decomposition

Uses `nalgebra`'s symmetric eigenvalue decomposition (Lanczos-based). Eigenvalues are sorted ascending; eigenvectors are stored as columns.

---

## The Math

### Cellular Sheaves

A cellular sheaf F on a graph G assigns:
- A vector space F(v) = ℝ^{d_v} to each vertex v ("stalk")
- A vector space F(e) = ℝ^{d_e} to each edge e ("edge stalk")
- A linear restriction map F_{v→e}: F(v) → F(e) for each incident pair

The space of 0-cochains C⁰(F) = ⊕_v F(v) (sections). The coboundary D: C⁰ → C¹ is defined by restriction maps.

### Sheaf Laplacian

L = DᵀD : C⁰(F) → C⁰(F)

Properties:
- L is symmetric positive semidefinite
- ker(L) = H⁰(F) (sheaf cohomology in degree 0)
- For constant sheaf d=1 on a connected graph: ker(L) = span(1), L = graph Laplacian

### Hodge Decomposition

C⁰(F) = ker(L) ⊕ im(Dᵀ)

The harmonic space ker(L) ≅ H⁰(F) captures globally consistent sections. The image space im(Dᵀ) captures "inconsistency." The decomposition is orthogonal.

### Sheaf Diffusion

The heat equation dx/dt = −Lx has solution x(t) = exp(−tL) x₀. As t → ∞, x(t) → P_harmonic · x₀ (projection onto harmonic space). Energy E(x) = xᵀLx decreases monotonically.

### Connection Laplacian

For a vector bundle with orthogonal connection maps ρ_{uv} on edges:

(L_conn)_{uu} = deg(u) · I
(L_conn)_{uv} = −ρ_{vu}

Trivial connections (ρ = I) recover the sheaf Laplacian. Rotation connections model SO(n)-synchronization.

### Persistent Sheaf Cohomology

Edges are sorted by weight to form a filtration. At each step, the sheaf Laplacian is computed and dim H⁰ is tracked. When two components merge, d bars die (where d is the stalk dimension). The resulting barcode captures the topological evolution of the sheaf.

---

## Test Suite

**55 tests** across all modules:

| module | tests |
|---|---|
| `graph` | 5 |
| `sheaf` | 3 |
| `laplacian` | 8 |
| `hodge` | 6 |
| `diffusion` | 7 |
| `connection` | 5 |
| `neural` | 6 |
| `synchronization` | 7 |
| `persistent` | 8 |

Run: `cargo test`

---

## License

MIT
