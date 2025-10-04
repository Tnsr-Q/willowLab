# Willow Lab Mathematical Constructs

## Step-1 invariants and topology batteries
- **Spectral vs. entanglement temperature duality**: the Spectral test computes the inverse curvature of the log-resolvent trace and compares it against the inverse slope of the entanglement entropy–energy curve near the JT≈1 window, enforcing a unit-slope, high-R² regression to certify the duality.【F:Willow lab setup.txt†L1182-L1276】
- **η-lock fault-tolerance signatures**: sliding windows demand consistent sign in η oscillations and frozen Chern parity, while spectral-flow crossings seed decoder priors that bias between parity and phase flips.【F:Willow lab setup.txt†L1285-L1347】
- **Wilson-loop geometry and residue atlas**: non-Abelian loops stack overlap matrices via SVD-aligned transport to report Abelian phases, eigenvalue spectra, and curvature logs; residues emerge from adjugate traces of (I−U) at near-zero eigenvalues to expose poles.【F:Willow lab setup.txt†L1364-L1448】
- **Nested T¹⁴ loops and black-hole potential**: seven-layer Kronecker stacking of Wilson curvatures yields a c₁₄ integer estimator, and residue-weighted 1/r² superposition builds the “black-hole” potential map across parameter grids.【F:Willow lab setup.txt†L1480-L1558】

## Parameter-space cartography analytics
- **Planned computations**: the cartography module scans two-control grids, extracts (I−U) poles and residues via the adjugate trick, sums gravitational potentials Φ(λ)=∑α |Rα|/(‖λ−λα‖ᵖ+ε) with p≈2, differentiates the phase field W=∇arg Tr(I−U)⁻¹, diagnoses EPs through det-winding, eigenvector conditioning, and branch-point tests, and finds saddles from Φ’s Hessian before feeding T¹⁴ workflows.【F:Willow lab ext 1.txt†L2-L178】
- **Residue-driven fields**: `black_hole_potential` discretely superposes high-residue sources, defaulting to the top 5% unless a custom mask is provided, while `phase_wind_field` gradients the argument of the resolvent trace to expose circulation of topological charge.【F:Willow lab ext 1.txt†L623-L758】
- **Determinant winding and saddles**: plaquette det-winding unwraps det(I−U) phases over 1×1 loops to accumulate 2π-normalized winding counts, and `hessian_saddles` tests the sign of Φ’s Hessian eigenvalues to isolate saddle basins.【F:Willow lab ext 1.txt†L760-L945】
- **Cancellation-safe resolvent magnitude**: reuse of the Step-1 surrogate switches between |1/(1−λ)| off the unit circle and 1/(2|sin(θ/2)|) on it, summing per grid point to avoid catastrophic cancellation.【F:Willow lab ext 1.txt†L947-L1018】
- **Exceptional-point candidates**: det-winding masks combine with large eigenvector condition numbers to flag EP plaquettes, creating a boolean lattice ready for contouring and downstream Wilson paths.【F:Willow lab ext 1.txt†L1020-L1069】

## Scaling notes
- The naive Φ superposition is O((NₓN_y)²); production guidance recommends truncating to the top residues, applying FFT-based Poisson solves for p=2 lattices, or tiling with halo blending to parallelize large parameter scans.【F:Willow lab ext 1.txt†L1570-L1628】
