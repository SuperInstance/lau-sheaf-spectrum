# lau-sheaf-spectrum

Spectral sheaf theory: the intersection of sheaf cohomology and spectral graph theory.

## Features

- **Sheaf Laplacian** — L = DᵀD where D is the coboundary operator
- **Hodge Decomposition** — Harmonic/exact splitting of sheaf cohomology
- **Spectral Gaps** — Eigenvalue analysis and synchronization conditions
- **Sheaf Diffusion** — Heat equation dx/dt = −Lx converging to harmonic sections
- **Connection Laplacian** — For vector bundles over graphs
- **Sheaf Neural Networks** — Single diffusion layer and multi-layer sheaf convolutions
- **Synchronization** — Multi-agent consensus as sheaf synchronization
- **Persistent Sheaf Cohomology** — Filtration → barcode of H⁰

## Usage

```rust
use lau_sheaf_spectrum::prelude::*;

// Create a graph and constant sheaf
let graph = Graph::complete(5);
let sheaf = Sheaf::constant(&graph, 2);

// Build sheaf Laplacian
let sl = SheafLaplacian::build(&sheaf, &graph);
println!("Spectral gap: {:?}", sl.spectral_gap());
println!("Kernel dim: {}", sl.kernel_dim());

// Hodge decomposition
let hodge = HodgeDecomposition::compute(&sheaf, &graph);
println!("Betti-0: {}", hodge.betti_0());

// Sheaf diffusion
let diff = SheafDiffusion::new(&sheaf, &graph);
let x0 = DVector::from_vec(vec![1.0; 10]);
let xt = diff.evolve(&x0, 10.0);
```

## License

MIT
