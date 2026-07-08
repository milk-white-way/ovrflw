#!/usr/bin/env python3
"""
plot_channel_profile.py
=======================
Read an AMReX pltResults plotfile and plot the streamwise velocity profile
Ux/Uin vs z at a specified (x, y) cross-section.

Usage:
    python3 plot_channel_profile.py [path/to/pltResults_dir] [options]

Without arguments, the script searches for the latest pltResults* directory
in ../../Exec/ (relative to this file).

Options:
    --x      X coordinate of the cross-section  (default: 5.0)
    --y      Y coordinate of the cross-section  (default: 0.5)
    --uin    Inflow velocity for normalization   (default: 1.0)
    --out    Output PNG path                     (default: next to plotfile)
"""

import os
import sys
import glob
import re
import argparse
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Plotfile discovery
# ---------------------------------------------------------------------------

def find_latest_plotfile(search_dir: str, prefix: str = "pltResults") -> str:
    dirs = sorted(glob.glob(os.path.join(search_dir, prefix + "*")))
    if not dirs:
        raise FileNotFoundError(f"No '{prefix}*' found in {search_dir}")
    return dirs[-1]


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def _next_nonempty(lines, idx):
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    return idx


def parse_amrex_header(pltdir: str) -> dict:
    with open(os.path.join(pltdir, "Header")) as f:
        lines = f.read().splitlines()

    idx = 0
    assert lines[idx].startswith("HyperCLaw"), f"Unknown format: {lines[0]}"
    idx += 1

    ncomp = int(lines[idx]); idx += 1
    comp_names = [lines[idx + i].strip() for i in range(ncomp)]; idx += ncomp

    spacedim  = int(lines[idx]); idx += 1
    time      = float(lines[idx]); idx += 1
    finest_lv = int(lines[idx]); idx += 1

    prob_lo = list(map(float, lines[idx].split())); idx += 1
    prob_hi = list(map(float, lines[idx].split())); idx += 1

    idx = _next_nonempty(lines, idx)
    # domain box: ((lo) (hi) (type))
    box_nums = list(map(int, re.findall(r"-?\d+", lines[idx]))); idx += 1
    domain_lo = box_nums[:spacedim]
    domain_hi = box_nums[spacedim:2 * spacedim]

    idx = _next_nonempty(lines, idx)
    idx += 1  # ref_ratio (skip)

    idx = _next_nonempty(lines, idx)
    dx = list(map(float, lines[idx].split())); idx += 1

    return {
        "ncomp":      ncomp,
        "comp_names": comp_names,
        "spacedim":   spacedim,
        "time":       time,
        "finest_lv":  finest_lv,
        "prob_lo":    prob_lo,
        "prob_hi":    prob_hi,
        "domain_lo":  domain_lo,
        "domain_hi":  domain_hi,
        "dx":         dx,
    }


# ---------------------------------------------------------------------------
# Cell_H parsing — returns list of (filename, byte_offset)
# ---------------------------------------------------------------------------

def parse_cell_h(pltdir: str):
    path = os.path.join(pltdir, "Level_0", "Cell_H")
    with open(path) as f:
        content = f.read()
    entries = []
    for m in re.finditer(r"FabOnDisk:\s+(\S+)\s+(\d+)", content):
        entries.append((m.group(1), int(m.group(2))))
    return entries


# ---------------------------------------------------------------------------
# FAB binary reader — returns the requested component as ndarray (nz, ny, nx)
# ---------------------------------------------------------------------------

def read_fab_component(fab_path: str, byte_offset: int,
                       spacedim: int, ncomp: int, comp_idx: int):
    with open(fab_path, "rb") as f:
        f.seek(byte_offset)

        # Read text header line
        hdr_bytes = b""
        while True:
            ch = f.read(1)
            if ch == b"\n" or ch == b"":
                break
            hdr_bytes += ch
        data_start = f.tell()
        hdr_str = hdr_bytes.decode("ascii", errors="replace")

        # Parse box and ncomp from FAB header
        # All integers in the header: format descriptors come first, then box coords, then ncomp
        all_ints = list(map(int, re.findall(r"-?\d+", hdr_str)))
        fab_ncomp = all_ints[-1]
        box_ints  = all_ints[-(1 + spacedim * 3):-1]
        blo = box_ints[:spacedim]
        bhi = box_ints[spacedim:2 * spacedim]

        nx = bhi[0] - blo[0] + 1
        ny = bhi[1] - blo[1] + 1
        nz = bhi[2] - blo[2] + 1 if spacedim == 3 else 1
        ntot = nx * ny * nz * fab_ncomp

        raw = np.frombuffer(f.read(ntot * 8), dtype=np.float64)

    # AMReX / Fortran layout: i fastest, then j, then k, comp slowest
    # Flat index = i + nx*(j + ny*(k + nz*comp))
    # → reshape (ncomp, nz, ny, nx) in C order gives arr[comp, k, j, i]
    arr = raw.reshape((fab_ncomp, nz, ny, nx))
    return arr[comp_idx], blo, bhi     # shape (nz, ny, nx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pltdir", nargs="?", default=None,
                        help="Path to pltResults directory")
    parser.add_argument("--x",   type=float, default=5.0,  help="x coordinate")
    parser.add_argument("--y",   type=float, default=0.5,  help="y coordinate")
    parser.add_argument("--uin", type=float, default=1.0,  help="Inflow velocity (normalisation)")
    parser.add_argument("--out", type=str,   default=None, help="Output PNG path")
    args = parser.parse_args()

    # Locate plotfile
    if args.pltdir:
        plt_path = args.pltdir
    else:
        script_dir   = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        exec_dir     = os.path.join(project_root, "Exec")
        plt_path = find_latest_plotfile(exec_dir, "pltResults")

    print(f"Plotfile : {plt_path}")

    # Parse header
    hdr = parse_amrex_header(plt_path)
    dx      = hdr["dx"]
    prob_lo = hdr["prob_lo"]
    dom_lo  = hdr["domain_lo"]
    dom_hi  = hdr["domain_hi"]
    spacedim = hdr["spacedim"]

    nx_dom = dom_hi[0] - dom_lo[0] + 1
    ny_dom = dom_hi[1] - dom_lo[1] + 1
    nz_dom = dom_hi[2] - dom_lo[2] + 1 if spacedim == 3 else 1

    print(f"Grid     : {nx_dom} x {ny_dom} x {nz_dom}  |  dx={dx[0]:.5f}")
    print(f"Time     : {hdr['time']:.6f}")
    print(f"Components: {hdr['comp_names']}")

    # Find U component
    names_low = [c.lower() for c in hdr["comp_names"]]
    if "u" in names_low:
        u_idx = names_low.index("u")
    else:
        raise ValueError(f"Cannot find 'U' in components: {hdr['comp_names']}")

    # Nearest cell indices to (x_target, y_target)
    def nearest_cell(coord, prob_lo_d, dx_d, lo, hi):
        i = int((coord - prob_lo_d) / dx_d)          # lower-left cell
        if i < lo: i = lo
        if i > hi: i = hi
        # pick between i and i+1 (whichever centre is closer)
        c0 = prob_lo_d + (i + 0.5) * dx_d
        c1 = prob_lo_d + (i + 1.5) * dx_d
        if i + 1 <= hi and abs(c1 - coord) < abs(c0 - coord):
            i += 1
        return i

    i_tgt = nearest_cell(args.x, prob_lo[0], dx[0], dom_lo[0], dom_hi[0])
    j_tgt = nearest_cell(args.y, prob_lo[1], dx[1], dom_lo[1], dom_hi[1])

    x_actual = prob_lo[0] + (i_tgt + 0.5) * dx[0]
    y_actual = prob_lo[1] + (j_tgt + 0.5) * dx[1]
    print(f"Slice    : i={i_tgt} (x={x_actual:.4f}),  j={j_tgt} (y={y_actual:.4f})")

    # Accumulate U profile along z
    u_line = np.full(nz_dom, np.nan)

    fab_entries = parse_cell_h(plt_path)
    for fab_rel, offset in fab_entries:
        fab_path = os.path.join(plt_path, "Level_0", fab_rel)

        comp_data, blo, bhi = read_fab_component(
            fab_path, offset, spacedim, hdr["ncomp"], u_idx
        )
        # comp_data shape: (nz_fab, ny_fab, nx_fab), indices local to blo

        if not (blo[0] <= i_tgt <= bhi[0] and blo[1] <= j_tgt <= bhi[1]):
            continue  # this FAB doesn't cover our (i, j)

        i_loc = i_tgt - blo[0]
        j_loc = j_tgt - blo[1]
        k_lo  = blo[2] if spacedim == 3 else 0
        k_hi  = bhi[2] if spacedim == 3 else 0

        for k_glob in range(k_lo, k_hi + 1):
            k_loc = k_glob - k_lo
            u_line[k_glob] = comp_data[k_loc, j_loc, i_loc]

    if np.any(np.isnan(u_line)):
        print("WARNING: some k-values were not filled — check FAB coverage")

    # z cell-centre coordinates
    z = np.array([prob_lo[2] + (k + 0.5) * dx[2] for k in range(nz_dom)])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(3.8, 5.0))

    ax.plot(u_line / args.uin, z,
            "s", color="black", markersize=4, markerfacecolor="none",
            linewidth=0.8, label=f"$t = {hdr['time']:.3f}$")

    ax.set_xlabel(r"$U_x \,/\, U_{in}$", fontsize=12)
    ax.set_ylabel(r"$L_z$",              fontsize=12)
    ax.set_title(
        f"Channel velocity profile\n"
        f"$x = {x_actual:.2f}$,  $y = {y_actual:.3f}$",
        fontsize=10,
    )
    ax.set_ylim(prob_lo[2], prob_lo[2] + nz_dom * dx[2])
    ax.set_xlim(left=0.0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.tick_params(direction="in", which="both")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=9, framealpha=0.8)

    fig.tight_layout()

    if args.out:
        out_path = args.out
    else:
        run_dir  = os.path.dirname(os.path.abspath(plt_path))
        out_path = os.path.join(run_dir, "channel_profile.png")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved    : {out_path}")


if __name__ == "__main__":
    main()
