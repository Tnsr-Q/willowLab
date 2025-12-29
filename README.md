# willowLab

![WillowLab Hero](image.png)

<div align="center">

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Nobel Validation](https://img.shields.io/badge/Validation-CR--5%20%7C%20CCC--2-ff69b4)](willowlab/spg.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A Computational Physics Framework for Stochastic Projective Gravity & Topological Phase Verification.**

[Features](#key-features) • [Architecture](#system-architecture) • [Quick Start](#quick-start) • [The Science](#the-science)

</div>

---

## 🪐 Overview

**willowLab** is a specialized research engine designed to ingest raw quantum simulation data (Floquet systems, Kitaev chains) and rigorously validate them against high-order theoretical theorems.

It serves as the bridge between raw eigen-data and "Nobel-ready" falsification, automating the detection of:
* **Spectral-Entanglement Duality** (Theorem B.1)
* **Exceptional Points & Residue Landscapes** (Theorem B.2)
* **Cosmic Ratchet & Dark Energy Thresholds** (SPG / CR-5)
* **Nested Wilson Loops & Higher-Form Topology** (Theorem B.3)

Whether you are analyzing a local `.npz` scan or a massive HDF5 cluster, willowLab normalizes the geometry into a unified schema for immediate topological cartography.

---

## ⚡ Key Features

### 1. The Trinity Engine
The core of the lab. It computes Step-1 invariants instantly upon data ingestion:
* **Cancellation-Safe Resolvents:** Accurate traces even near $| \lambda | \approx 1$.
* **$\eta$-Lock Detection:** Identifies protected topological windows via mod-2 Chern parity.
* **Duality Checks:** Correlates spectral temperature with entanglement thermodynamics.

### 2. Parameter Cartography
Turn abstract matrices into navigable maps.
* **Black Hole Potentials:** Visualizes gravitational potential $\Phi(\lambda)$ derived from residue superposition.
* **Saddle Detection:** Automatically flags mountain pass geometries in the phase space.
* **Wind Fields:** Computes phase winding $\nabla \arg \text{Tr}(I-U)^{-1}$ to isolate topological charges.

### 3. SPG & Nobel Validation Suite
A production-grade falsification runner that tests your data against physical reality:
* **CR-5 Protocol:** Checks for FRW-Radar acceleration thresholds ($AP' < -1/3$).
* **Pantheon+ Compliance:** Ensures operational curvature $|\Omega_{op}| < 0.02$.
* **32-Cell Classification:** Categorizes Floquet unitary operators into robust bit-packed geometric cells.

---

## 🏗 System Architecture

The pipeline is designed for robustness—from "Zip to Truth."

![WillowLab Data Flow](https://raw.githubusercontent.com/Tnsr-Q/willowLab/main/assets/data_flow_diagram.png)
*(Note: Upload the second blue schematic image to an `/assets` folder and link it here)*

### Directory Structure

```text
willowLab/
├── configs/             # YAML orchestration for validation runs
├── demos/               # Visualization scripts (Cartography, Potentials)
├── willowlab/
│   ├── ingest/          # "Zip-to-Willow" normalization pipeline
│   ├── geometry.py      # Non-Abelian Wilson loops & Residue Atlas
│   ├── spg.py           # Stochastic Projective Gravity (CR-5/CR-4)
│   ├── trinity.py       # Step-1 Invariant computer
│   ├── cartography.py   # Scalar fields & Pole detection
│   └── tests/           # Nobel Validation Suites (T_Nobel)
└── README.md

🚀 Quick Start
Installation
Clone the repository and install the Conda environment (Python 3.11 recommended).
git clone [https://github.com/Tnsr-Q/willowLab.git](https://github.com/Tnsr-Q/willowLab.git)
cd willowLab
conda env create -f environment.yml
conda activate willowlab

Running a Nobel Validation
To run the full suite of theorems against a dataset:
# Run the CLI with the "nobel_validation" command
python -m willowlab.cli nobel_validation "reports/my_submission_report.json"

Generating Cartography Maps
Visualize the "Black Hole Potential" and residue landscape of your system:
from willowlab.cartography import poles_and_residues_on_grid, black_hole_potential
import numpy as np
import matplotlib.pyplot as plt

# 1. Load your Unitary Grid
Ugrid = np.load("data/floquet_scan.npz")["Ugrid"]

# 2. Extract Residues (Poles)
atlas = poles_and_residues_on_grid(Ugrid)

# 3. Compute Gravitational Potential
Phi = black_hole_potential(atlas["residue_score"])

# 4. Visualize
plt.imshow(Phi, origin="lower", cmap="magma")
plt.title("Gravitational Potential $\Phi(\lambda)$")
plt.show()

🧬 The Science
Stochastic Projective Gravity (SPG)
willowLab implements the CR-5 criteria, mapping the velocity of the order parameter \xi to the FRW equation of state:
$$ AP' = \tanh\left( \frac{\kappa \cdot \dot{\xi}}{\gamma} \right) $$
A valid geometry must exhibit a "Dark Energy" crossing (AP' < -1/3) at the topological transition point JT^* \approx 1.0.
Higher-Form Topology (T^{14})
For advanced protection, the t14.py module computes invariants over nested 7-torus loops:
$$ c_{14} = \frac{1}{(2\pi)^7} \text{Tr} \left( \bigotimes_{i=1}^7 W_i \right) $$
🤝 Contributing
We welcome contributions to the Residue Atlas and Ingest Sniffers.
 * Fork the repo.
 * Create your feature branch (git checkout -b feature/AmazingPhysics).
 * Commit your changes.
 * Open a Pull Request.
📄 License
Distributed under the MIT License. See LICENSE for more information.
<div align="center">
<sub>Built by Tnsr-Q for the advancement of Geometric AI and Quantum Simulation.</sub>
</div>

