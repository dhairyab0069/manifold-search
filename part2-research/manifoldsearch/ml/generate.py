"""Generate labelled datasets: one .npz and .json per mesh.

    python -m manifoldsearch.ml.generate experiments/configs/ml_adaptive.yaml --seed 0 --out results/ml/data [--only grid_sphere:32,...]
"""
from __future__ import annotations

import argparse
import os

import yaml

from .dataset import generate_mesh, mesh_list, save_mesh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    ap.add_argument("--only", default=None, help="comma list family:size to restrict (default: all in the config)")
    ap.add_argument("--queries", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    todo = mesh_list(cfg)
    if args.only:
        keep = {(a.split(":")[0], int(a.split(":")[1])) for a in args.only.split(",")}
        todo = [m for m in todo if m in keep]
    for family, size in todo:
        data, meta = generate_mesh(family, size, args.seed, args.queries or cfg["queries"], cfg["targets_per_source"], args.procs,
                                   cfg.get("patches_per_sqrt", 1.0))
        save_mesh(args.out, data, meta)
        print(meta, flush=True)


if __name__ == "__main__":
    main()
