"""
OvrFlw benchmark comparison script.

Cases:
  tgv  -- 2D Taylor-Green Vortex vs. analytical solution
  ldc  -- 2D Lid-Driven Cavity vs. Ghia et al. (1982) at Re=1000

Usage:
  python compare.py --case tgv --pltfile <path>  [--plot] [--tol <float>]
  python compare.py --case ldc --pltfile <path>  [--plot] [--tol <float>]
  python compare.py --case tgv --fextract <fextract_exe> --pltfile <path>
"""

import argparse
import csv
import math
import os
import sys
import subprocess
import tempfile

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Default tolerances ──────────────────────────────────────────────────────
DEFAULT_TOL = {"tgv": 1e-1, "ldc": 5e-2}

# ── Ghia reference data ─────────────────────────────────────────────────────
GHIA_RE_COLS = {100: "Re100", 400: "Re400", 1000: "Re1000",
                3200: "Re3200", 5000: "Re5000", 7500: "Re7500", 10000: "Re10000"}


def load_ghia(csv_path, re):
    col = GHIA_RE_COLS[re]
    with open(csv_path) as f:
        reader = csv.DictReader(row for row in f if not row.startswith('#'))
        rows = list(reader)
        coord_col = reader.fieldnames[0]
    coord = np.array([float(r[coord_col].strip()) for r in rows])
    vel   = np.array([float(r[col].strip())       for r in rows])
    return coord, vel


# ── fextract helper ──────────────────────────────────────────────────────────
def run_fextract(fextract, pltfile, direction, fixed_axis, fixed_coord, variable):
    """Run fextract and return (coord_array, vel_array)."""
    axis_flag = {0: "-x", 1: "-y", 2: "-z"}
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        outfile = f.name
    cmd = [
        fextract,
        "-d", str(direction),
        axis_flag[fixed_axis], str(fixed_coord),
        "-v", variable,
        "-csv",
        "-s", outfile,
        pltfile,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("fextract error:", result.stderr)
        sys.exit(1)
    data = np.genfromtxt(outfile, delimiter=",", names=True, comments="#")
    os.unlink(outfile)
    # fextract CSV columns: x, y, z, <variable>
    coord_col = ["x", "y", "z"][direction]
    return data[coord_col], data[variable]


# ── yt plotfile reader ───────────────────────────────────────────────────────
def load_centerline_yt(pltfile, direction, fixed_coords, variable, nsamples=200):
    """
    Extract a 1D profile from an AMReX plotfile using yt.
    direction: axis along which we sample (0=x, 1=y, 2=z)
    fixed_coords: dict of the other two axes, e.g. {"x": 0.5, "z": 0.5}
    """
    import yt
    yt.set_log_level("error")
    ds = yt.load(pltfile)
    lo = ds.domain_left_edge.v
    hi = ds.domain_right_edge.v
    axes = ["x", "y", "z"]

    start = lo.copy()
    end   = hi.copy()
    for ax, val in fixed_coords.items():
        i = axes.index(ax)
        start[i] = val
        end[i]   = val

    ray = ds.ray(start, end)
    order = np.argsort(ray["t"])
    coord = ray[axes[direction]].v[order]
    vel   = ray[("boxlib", variable)].v[order]
    return coord, vel


# ── TGV analytical solution ──────────────────────────────────────────────────
def tgv_analytical(x, y, t, nu):
    u =  np.sin(x) * np.cos(y) * np.exp(-2.0 * nu * t)
    v = -np.cos(x) * np.sin(y) * np.exp(-2.0 * nu * t)
    return u, v


def l2_error(numerical, reference):
    return np.sqrt(np.mean((numerical - reference) ** 2))


# ── TGV test ─────────────────────────────────────────────────────────────────
def test_tgv(pltfile, fextract, tol, plot):
    print("\n  [TGV] Loading plotfile:", pltfile)

    import yt
    yt.set_log_level("error")
    ds = yt.load(pltfile)
    t  = float(ds.current_time)
    nu = 1.0  # Re=1 => nu=1

    lo = ds.domain_left_edge.v
    hi = ds.domain_right_edge.v
    is_3d = ds.dimensionality == 3

    # Extract u along y at x=pi
    x_mid = math.pi
    fixed_u = {"x": x_mid, "z": 0.5 * (lo[2] + hi[2])} if is_3d else {"x": x_mid}
    coord_y, u_num = load_centerline_yt(
        pltfile, direction=1, fixed_coords=fixed_u, variable="U")

    u_ana, _ = tgv_analytical(x_mid, coord_y, t, nu)
    err_u = l2_error(u_num, u_ana)

    # Extract v along x at y=pi
    y_mid = math.pi
    fixed_v = {"y": y_mid, "z": 0.5 * (lo[2] + hi[2])} if is_3d else {"y": y_mid}
    coord_x, v_num = load_centerline_yt(
        pltfile, direction=0, fixed_coords=fixed_v, variable="V")

    _, v_ana = tgv_analytical(coord_x, y_mid, t, nu)
    err_v = l2_error(v_num, v_ana)

    print(f"  [TGV] t = {t:.4f},  nu = {nu}")
    print(f"  [TGV] L2 error  u : {err_u:.6e}  (tol = {tol:.1e})")
    print(f"  [TGV] L2 error  v : {err_v:.6e}  (tol = {tol:.1e})")

    if plot:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(coord_y, u_num, "-",  color="#3a6b2f", lw=1.5, label="OvrFlw")
        ax1.plot(coord_y, u_ana, "--", color="#888888",          label="Analytical")
        ax1.set_xlabel("y");  ax1.set_ylabel("u");  ax1.set_title(f"TGV u-velocity  t={t:.3f}")
        ax1.legend()
        ax2.plot(coord_x, v_num, "-",  color="#3a6b2f", lw=1.5, label="OvrFlw")
        ax2.plot(coord_x, v_ana, "--", color="#888888",          label="Analytical")
        ax2.set_xlabel("x");  ax2.set_ylabel("v");  ax2.set_title(f"TGV v-velocity  t={t:.3f}")
        ax2.legend()
        plt.tight_layout()
        plt.savefig("tgv_comparison.png", dpi=150)
        print("  [TGV] Plot saved to tgv_comparison.png")

    passed = (err_u < tol) and (err_v < tol)
    return passed, max(err_u, err_v)


# ── LDC test ─────────────────────────────────────────────────────────────────
def test_ldc(pltfile, fextract, tol, re, plot):
    print(f"\n  [LDC] Loading plotfile: {pltfile}  (Re={re})")

    ghia_u_csv = os.path.join(SCRIPT_DIR, "ghia_u.csv")
    ghia_v_csv = os.path.join(SCRIPT_DIR, "ghia_v.csv")
    ghia_y, ghia_u = load_ghia(ghia_u_csv, re)
    ghia_x, ghia_v = load_ghia(ghia_v_csv, re)

    import yt
    yt.set_log_level("error")
    ds = yt.load(pltfile)
    lo = ds.domain_left_edge.v
    hi = ds.domain_right_edge.v
    is_3d = ds.dimensionality == 3
    x_mid = 0.5 * (lo[0] + hi[0])
    y_mid = 0.5 * (lo[1] + hi[1])

    # u along vertical centerline (x=0.5)
    fixed_u = {"x": x_mid, "z": 0.5 * (lo[2] + hi[2])} if is_3d else {"x": x_mid}
    coord_y, u_num = load_centerline_yt(
        pltfile, direction=1, fixed_coords=fixed_u, variable="U")

    # v along horizontal centerline (y=0.5)
    fixed_v = {"y": y_mid, "z": 0.5 * (lo[2] + hi[2])} if is_3d else {"y": y_mid}
    coord_x, v_num = load_centerline_yt(
        pltfile, direction=0, fixed_coords=fixed_v, variable="V")

    # Normalise coords to [0,1] for comparison with Ghia
    Lx = hi[0] - lo[0];  Ly = hi[1] - lo[1]
    coord_y_norm = (coord_y - lo[1]) / Ly
    coord_x_norm = (coord_x - lo[0]) / Lx

    # Interpolate Ghia data onto numerical grid
    ghia_u_interp = np.interp(coord_y_norm, ghia_y[::-1], ghia_u[::-1])
    ghia_v_interp = np.interp(coord_x_norm, ghia_x[::-1], ghia_v[::-1])

    err_u = l2_error(u_num, ghia_u_interp)
    err_v = l2_error(v_num, ghia_v_interp)

    print(f"  [LDC] L2 error  u (vs Ghia) : {err_u:.6e}  (tol = {tol:.1e})")
    print(f"  [LDC] L2 error  v (vs Ghia) : {err_v:.6e}  (tol = {tol:.1e})")

    if plot:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(u_num, coord_y_norm, "-", color="#3a6b2f", lw=1.5, label="OvrFlw")
        ax1.scatter(ghia_u, ghia_y, marker="*", color="#ff6eb4", s=80, label="Ghia (1982)")
        ax1.set_xlabel("u");  ax1.set_ylabel("y");  ax1.set_title(f"LDC u-velocity  Re={re}")
        ax1.legend()
        ax2.plot(coord_x_norm, v_num, "-", color="#3a6b2f", lw=1.5, label="OvrFlw")
        ax2.scatter(ghia_x, ghia_v, marker="*", color="#ff6eb4", s=80, label="Ghia (1982)")
        ax2.set_xlabel("x");  ax2.set_ylabel("v");  ax2.set_title(f"LDC v-velocity  Re={re}")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(f"ldc_re{re}_comparison.png", dpi=150)
        print(f"  [LDC] Plot saved to ldc_re{re}_comparison.png")

    passed = (err_u < tol) and (err_v < tol)
    return passed, max(err_u, err_v)


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="OvrFlw benchmark comparison")
    parser.add_argument("--case",     required=True, choices=["tgv", "ldc"])
    parser.add_argument("--pltfile",  required=True, help="AMReX plotfile directory")
    parser.add_argument("--fextract", default="fextract.gnu.ex",
                        help="Path to fextract executable (optional, yt used by default)")
    parser.add_argument("--re",       type=int, default=1000,
                        help="Reynolds number for LDC (default: 1000)")
    parser.add_argument("--tol",      type=float, default=None,
                        help="L2 error tolerance for pass/fail")
    parser.add_argument("--plot",     action="store_true",
                        help="Save comparison plots as PNG")
    args = parser.parse_args()

    tol = args.tol if args.tol is not None else DEFAULT_TOL[args.case]

    if not os.path.isdir(args.pltfile):
        print(f"ERROR: plotfile not found: {args.pltfile}")
        sys.exit(1)

    if args.case == "tgv":
        passed, err = test_tgv(args.pltfile, args.fextract, tol, args.plot)
    else:
        passed, err = test_ldc(args.pltfile, args.fextract, tol, args.re, args.plot)

    print()
    if passed:
        print(f"  PASS  {args.case.upper()}  (max L2 error = {err:.3e} < tol = {tol:.1e})")
    else:
        print(f"  FAIL  {args.case.upper()}  (max L2 error = {err:.3e} >= tol = {tol:.1e})")
    print()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
