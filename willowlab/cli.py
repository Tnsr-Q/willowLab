import json
import pathlib

try:  # pragma: no cover - optional dependency
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - use shim for tests
    from . import _numpy_shim as np  # type: ignore

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback when PyYAML absent
    yaml = None  # type: ignore[assignment]

def _load_config(config_path):
    if yaml is None:
        raise RuntimeError("PyYAML is required for this command")
    return yaml.safe_load(open(config_path))

def run(config_path):
    from .geometry import residue_atlas
    from .io import load_willow
    from .tests.t_eta_lock import decoder_priors_from_crossings, eta_lock_windows
    from .tests.t_spec_ent import test_duality
    from .trinity import WillowTrinityStep1

    cfg = _load_config(config_path)
    ds = load_willow(cfg["dataset"])

    # Step-1 invariants
    tri = WillowTrinityStep1(ds); inv = tri.compute_all(
        jt_star=cfg.get("jt_star",1.0),
        window=cfg.get("window",0.05)
    )
    out = {"invariants": inv}

    # Tests (only run when inputs present)
    if "t_spec_ent" in cfg["tests"] and ds.resolvent_trace is not None and ds.entropy is not None:
        out["t_spec_ent"] = test_duality(ds)

    if "t_eta_lock" in cfg["tests"] and ds.eta_oscillations is not None and ds.chern_mod2 is not None:
        locks = eta_lock_windows(ds.eta_oscillations, ds.chern_mod2)
        priors = decoder_priors_from_crossings(ds.spectral_flow_crossings or [])
        out["t_eta_lock"] = {"lock_windows": locks.tolist(), "decoder_priors": priors}

    if "t_geometry" in cfg["tests"] and ds.floquet_operators is not None:
        out["t_geometry"] = residue_atlas(ds.floquet_operators)

    art = pathlib.Path(cfg["artifacts_dir"]); art.mkdir(parents=True, exist_ok=True)
    (art/"summary.json").write_text(json.dumps(out, indent=2, default=lambda x: x if
                                               isinstance(x, (int, float, str)) else str(x)))
    print(f"Wrote {art/'summary.json'}")


def run_resolvent(config_path):
    from .io import load_willow
    from .resolvent import resolvent_scan, spectral_temperature, validate_theorem_b1

    """
    Run resolvent witness analysis on Willow dataset.

    Config keys:
        dataset: path to .npz file
        jt_star: peak location (default 1.0)
        window: analysis window (default 0.05)
        artifacts_dir: output directory
    """
    cfg = _load_config(config_path)
    ds = load_willow(cfg["dataset"])

    if ds.floquet_eigenvalues is None:
        print("Error: No eigenvalues in dataset")
        return

    # Run resolvent scan
    result = resolvent_scan(
        ds.floquet_eigenvalues,
        ds.JT_scan_points,
        align_phase=True
    )

    out = {
        'peak_jt': result['peak_jt'],
        'peak_r_op': result['peak_r_op'],
        'expected_jt': cfg.get('jt_star', 1.0),
        'peak_match': abs(result['peak_jt'] - cfg.get('jt_star', 1.0)) < cfg.get('window', 0.05)
    }

    # Theorem B.1 test if entropy data present
    if ds.entropy is not None and ds.effective_energy is not None:
        duality = validate_theorem_b1(
            result['trace_abs'],
            ds.JT_scan_points,
            ds.entropy,
            ds.effective_energy,
            window=cfg.get('window', 0.05)
        )
        out['theorem_b1'] = duality

    # Write output
    art = pathlib.Path(cfg.get("artifacts_dir", "./artifacts"))
    art.mkdir(parents=True, exist_ok=True)

    (art/"resolvent_results.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {art/'resolvent_results.json'}")
    print(f"Peak at JT={out['peak_jt']:.3f}, R_op={out['peak_r_op']:.3f}")

    if 'theorem_b1' in out:
        print(f"Theorem B.1: slope={out['theorem_b1']['slope']:.3f}, pass={out['theorem_b1']['passed']}")


def run_spectral_flow(config_path):
    from .io import load_willow
    from .spectral_flow import berry_phase_loop, chern_number, validate_theorem_b4

    """
    Run spectral flow topology analysis.

    Config keys:
        dataset: path to .npz file
        loop_mode: 'single' or 'nested' (for c₁₄)
        artifacts_dir: output directory
    """
    cfg = _load_config(config_path)
    ds = load_willow(cfg["dataset"])

    if ds.floquet_eigenvectors is None:
        print("Error: No eigenvectors in dataset")
        return

    # Extract closed loop from eigenvectors
    evecs_loop = list(ds.floquet_eigenvectors)
    evecs_loop.append(evecs_loop[0])  # Close loop

    # Compute Berry phases
    berry_phases = berry_phase_loop(evecs_loop)
    C = chern_number(berry_phases)

    out = {
        'berry_phases': berry_phases.tolist(),
        'chern_number': int(C),
        'chern_mod2': int(C % 2)
    }

    # Theorem B.4 test if η data present
    if ds.eta_oscillations is not None and ds.chern_mod2 is not None:
        eta_test = validate_theorem_b4(berry_phases, ds.eta_oscillations, ds.chern_mod2)
        out['theorem_b4'] = eta_test

    # Write output
    art = pathlib.Path(cfg.get("artifacts_dir", "./artifacts"))
    art.mkdir(parents=True, exist_ok=True)

    (art/"spectral_flow_results.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {art/'spectral_flow_results.json'}")
    print(f"Chern number: {C}")

    if 'theorem_b4' in out:
        print(f"Theorem B.4: agreement={out['theorem_b4']['agreement_rate']:.1%}, pass={out['theorem_b4']['passed']}")


def run_disorder(config_path):
    from .disorder import disorder_scan, optimal_disorder
    from .io import load_willow

    """
    Run disorder sharpening analysis.

    Config keys:
        dataset: path to .npz file
        delta_values: array of disorder strengths (default [0.0, 0.05, 0.1, 0.15, 0.2])
        n_realizations: disorder realizations per delta (default 5)
        artifacts_dir: output directory
    """
    cfg = _load_config(config_path)
    ds = load_willow(cfg["dataset"])

    if ds.floquet_eigenvalues is None:
        print("Error: No eigenvalues in dataset")
        return

    delta_values = np.array(cfg.get('delta_values', [0.0, 0.05, 0.1, 0.15, 0.2]))
    n_realizations = cfg.get('n_realizations', 5)

    # Run disorder scan
    results = disorder_scan(
        ds.floquet_eigenvalues,
        ds.JT_scan_points,
        delta_values,
        n_realizations=n_realizations
    )

    optimal = optimal_disorder(results)

    out = {
        'disorder_scan': results,
        'optimal_delta': optimal['optimal_delta'],
        'enhancement_factor': optimal['enhancement_factor']
    }

    # Write output
    art = pathlib.Path(cfg.get("artifacts_dir", "./artifacts"))
    art.mkdir(parents=True, exist_ok=True)

    (art/"disorder_results.json").write_text(
        json.dumps(out, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    )
    print(f"Wrote {art/'disorder_results.json'}")
    print(f"Optimal δ={optimal['optimal_delta']:.2f}, enhancement={optimal['enhancement_factor']:.2f}x")


def run_nobel_validation_cli(report_path: str = "nobel_validation_report.json"):
    """Run the consolidated Nobel validation workflow."""

    from .tests.t_nobel_validation import execute_nobel_validation

    print("🏆 RUNNING NOBEL-LEVEL VALIDATION SUITE")
    print("This validates all critical falsification criteria...")

    try:
        report = execute_nobel_validation(report_path)
    except AssertionError as exc:
        print(f"❌ VALIDATION FAILED: {exc}")
        return

    print("✅ VALIDATION COMPLETE")
    print(f"Report saved: {pathlib.Path(report_path).resolve()}")
    if report.get("overall_conclusion") == "ALL CRITICAL THEOREMS VALIDATED":
        print("🎉 READY FOR NOBEL SUBMISSION!")
    else:
        print("❌ FRAMEWORK FALSIFIED - REQUIRES REVISION")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m willowlab.cli <command> [config.yaml]")
        sys.exit(1)

    command = sys.argv[1]
    requires_config = {"trinity", "resolvent", "spectral-flow", "disorder"}

    if command in requires_config:
        if len(sys.argv) < 3:
            print(f"Command '{command}' requires a configuration file")
            sys.exit(1)
        config_path = sys.argv[2]

        if command == "trinity":
            run(config_path)
        elif command == "resolvent":
            run_resolvent(config_path)
        elif command == "spectral-flow":
            run_spectral_flow(config_path)
        elif command == "disorder":
            run_disorder(config_path)
    elif command == "nobel_validation":
        report_path = sys.argv[2] if len(sys.argv) > 2 else "nobel_validation_report.json"
        run_nobel_validation_cli(report_path)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
