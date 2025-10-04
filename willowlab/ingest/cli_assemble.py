# willowlab/ingest/cli_assemble.py
import argparse, glob
from .assemble_workflow import assemble_from_zips

def main():
    ap = argparse.ArgumentParser(description="Assemble Willow bundles from zips + expanded dirs")
    ap.add_argument("--zip-glob", required=True, help="Glob for zip files (e.g., '/data/*.zip')")
    ap.add_argument("--expanded-dirs", nargs="*", default=[], help="Dirs expanded next to each zip")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--merge-policy", choices=["auto","prefer_operator","prefer_supplied"], default="auto",
                    help="Resolve eigenpair conflicts")
    ap.add_argument("--cache-root", default=None, help="Root directory for incremental cache DB")
    args = ap.parse_args()
    zip_paths = glob.glob(args.zip_glob)
    summary_path = assemble_from_zips(zip_paths, args.out,
                                      expanded_dirs=args.expanded_dirs,
                                      merge_policy=args.merge_policy,
                                      cache_root=args.cache_root)
    print(f"Wrote {summary_path}")

if __name__ == "__main__":
    main()
