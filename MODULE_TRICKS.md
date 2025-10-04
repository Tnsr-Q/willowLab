# Willow Lab Module Integration & Operational Tricks

## Project skeleton and environment
- Standardize on a reproducible Conda stack (Python 3.11 plus NumPy, SciPy, pandas, h5py, numba, PyYAML, Matplotlib, NetworkX, pytest).【F:Willow lab setup.txt†L205-L324】
- Organize the repository so ingestion, caching, Trinity analysis, tests, geometry, T¹⁴ logic, and CLI live in predictable modules under `willowlab/` with YAML configs for orchestration.

## Schema-driven ingestion core
- The immutable `WillowDataset` dataclass unifies JT scan points, Floquet eigen-data, resolvent traces, entropy, η oscillations, Chern parity, spectral crossings, and optional overlap matrices, validating shapes before memoized computations populate caches.【F:Willow lab setup.txt†L324-L512】

## Trinity and test orchestration
- The CLI loader hydrates a Willow bundle, runs the Step-1 Trinity invariants (`WillowTrinityStep1.compute_all`) with configurable JT anchor and window, conditionally executes Spectral duality, η-lock, and geometry batteries when data is present, and records JSON summaries to an artifacts directory.【F:Willow lab setup.txt†L1600-L1718】

## Zip-to-Willow conveyor
- Stage A triage inspects zip members by magic bytes, records a manifest, and opportunistically lifts JT grids or eigenpairs, while Stage B extractors reconcile partial payloads into complete Willow bundles before normalization and persistence to hashed `.npz`/`.h5` artifacts with provenance manifests.【F:Willow lab Injest.txt†L1-L200】【F:Willow lab Injest.txt†L520-L678】

## Incremental caching and HDF5 probing
- `DirCache` tracks size/mtime and SHA256 digests per ingestion root to skip unchanged files, and the one-level HDF5 walker samples root and child groups within memory budgets so foreign layouts still surface key tensors.【F:Willow lab Helpers.txt†L1-L520】

## Eigenpair reconciliation and CLI ergonomics
- Merge policy helpers derive eigenpairs from operators when needed, resolve conflicts across supplied data, and expose an `--merge-policy` flag (`auto`, `prefer_operator`, `prefer_supplied`) alongside `--cache-root` so the assembly CLI can steer conflicts and reuse cached results; usage recipes show first-run and incremental invocations plus guardrails for cache validity and walker budgets.【F:Willow lab Helpers.txt†L720-L1760】
