import yaml, json, pathlib
from .io import load_willow
from .trinity import WillowTrinityStep1
from .tests.t_spec_ent import test_duality
from .tests.t_eta_lock import eta_lock_windows, decoder_priors_from_crossings
from .geometry import non_abelian_wilson_loop, residue_atlas

def run(config_path):
    cfg = yaml.safe_load(open(config_path))
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
