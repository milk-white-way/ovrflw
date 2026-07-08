#!/usr/bin/env python3
"""
plot_channel_profiles.py
========================
Read one or more AMReX pltResults plotfiles and overlay their streamwise
velocity profiles (Ux/Uin vs z) at a specified (x, y) cross-section.
Reproduces the style of Figure 1 in the channel-flow benchmark paper.

Usage:
    python3 plot_channel_profiles.py \\
        path/to/pltResults_dir  "Re=0.1 typical"  \\
        path/to/pltResults_dir2 "Re=0.01 typical" \\
        [--x 5.0] [--y 0.5] [--uin 1.0] [--out profile_comparison.png]

Each (dir, label) pair is provided as two consecutive positional arguments.
"""

import os
import sys
import re
import struct
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── AMReX plotfile parser (same core as plot_channel_profile.py) ────────────

def _next_nonempty(lines, idx):
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    return idx


def parse_amrex_header(pltdir):
    with open(os.path.join(pltdir, "Header")) as f:
        lines = f.read().splitlines()

    idx = 0
    assert lines[idx].startswith("HyperCLaw"), f"Unknown format: {lines[0]}"
    idx += 1

    ncomp      = int(lines[idx]); idx += 1
    comp_names = [lines[idx + i].strip() for i in range(ncomp)]; idx += ncomp
    spacedim   = int(lines[idx]); idx += 1
    time       = float(lines[idx]); idx += 1
    _finest_lv = int(lines[idx]); idx += 1
    prob_lo    = list(map(float, lines[idx].split())); idx += 1
    prob_hi    = list(map(float, lines[idx].split())); idx += 1

    idx = _next_nonempty(lines, idx)
    box_nums = list(map(int, re.findall(r"-?\d+", lines[idx]))); idx += 1
    domain_lo = box_nums[:spacedim]
    domain_hi = box_nums[spacedim:2 * spacedim]

    idx = _next_nonempty(lines, idx)
    idx += 1  # ref_ratio
    idx = _next_nonempty(lines, idx)
    dx = list(map(float, lines[idx].split()))

    return {
        "ncomp":      ncomp,
        "comp_names": comp_names,
        "spacedim":   spacedim,
        "time":       time,
        "prob_lo":    prob_lo,
        "domain_lo":  domain_lo,
        "domain_hi":  domain_hi,
        "dx":         dx,
    }


def parse_cell_h(pltdir):
    with open(os.path.join(pltdir, "Level_0", "Cell_H")) as f:
        content = f.read()
    entries = []
    for m in re.finditer(r"FabOnDisk:\s+(\S+)\s+(\d+)", content):
        entries.append((m.group(1), int(m.group(2))))
    return entries


def read_fab_component(fab_path, byte_offset, spacedim, ncomp, comp_idx):
    with open(fab_path, "rb") as f:
        f.seek(byte_offset)
        hdr_bytes = b""
        while True:
            ch = f.read(1)
            if ch == b"\n" or ch == b"":
                break
            hdr_bytes += ch
        hdr_str = hdr_bytes.decode("ascii", errors="replace")
        all_ints  = list(map(int, re.findall(r"-?\d+", hdr_str)))
        fab_ncomp = all_ints[-1]
        box_ints  = all_ints[-(1 + spacedim * 3):-1]
        blo = box_ints[:spacedim]
        bhi = box_ints[spacedim:2 * spacedim]
        nx = bhi[0] - blo[0] + 1
        ny = bhi[1] - blo[1] + 1
        nz = bhi[2] - blo[2] + 1 if spacedim == 3 else 1
        raw = np.frombuffer(f.read(nx * ny * nz * fab_ncomp * 8), dtype=np.float64)

    arr = raw.reshape((fab_ncomp, nz, ny, nx))
    return arr[comp_idx], blo, bhi


def nearest_cell(coord, prob_lo_d, dx_d, lo, hi):
    i = int((coord - prob_lo_d) / dx_d)
    i = max(lo, min(hi, i))
    c0 = prob_lo_d + (i + 0.5) * dx_d
    c1 = prob_lo_d + (i + 1.5) * dx_d
    if i + 1 <= hi and abs(c1 - coord) < abs(c0 - coord):
        i += 1
    return i


def extract_u_profile(pltdir, x_tgt, y_tgt):
    hdr      = parse_amrex_header(pltdir)
    dx       = hdr["dx"]
    prob_lo  = hdr["prob_lo"]
    dom_lo   = hdr["domain_lo"]
    dom_hi   = hdr["domain_hi"]
    spacedim = hdr["spacedim"]

    nz_dom = dom_hi[2] - dom_lo[2] + 1 if spacedim == 3 else 1

    names_low = [c.lower() for c in hdr["comp_names"]]
    if "u" not in names_low:
        raise ValueError(f"'U' not in components: {hdr['comp_names']}")
    u_idx = names_low.index("u")

    i_tgt = nearest_cell(x_tgt, prob_lo[0], dx[0], dom_lo[0], dom_hi[0])
    j_tgt = nearest_cell(y_tgt, prob_lo[1], dx[1], dom_lo[1], dom_hi[1])

    u_line = np.full(nz_dom, np.nan)

    for fab_rel, offset in parse_cell_h(pltdir):
        fab_path = os.path.join(pltdir, "Level_0", fab_rel)
        comp_data, blo, bhi = read_fab_component(
            fab_path, offset, spacedim, hdr["ncomp"], u_idx)

        if not (blo[0] <= i_tgt <= bhi[0] and blo[1] <= j_tgt <= bhi[1]):
            continue

        i_loc = i_tgt - blo[0]
        j_loc = j_tgt - blo[1]
        k_lo  = blo[2] if spacedim == 3 else 0
        k_hi  = bhi[2] if spacedim == 3 else 0
        for k_glob in range(k_lo, k_hi + 1):
            u_line[k_glob] = comp_data[k_glob - k_lo, j_loc, i_loc]

    z = np.array([prob_lo[2] + (k + 0.5) * dx[2] for k in range(nz_dom)])
    return z, u_line, hdr["time"]


# ── Markers / colors cycling (matches paper style) ──────────────────────────

_MARKERS = ["s", "o", "^", "D", "v", "P", "X", "*"]
_COLORS  = ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728",
            "#8c564b", "#e377c2", "#7f7f7f"]


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    raw_args = sys.argv[1:]

    # Split positional pairs from optional flags
    pos_args  = []
    opt_args  = []
    i = 0
    while i < len(raw_args):
        if raw_args[i].startswith("--"):
            opt_args += raw_args[i:i+2]
            i += 2
        else:
            pos_args.append(raw_args[i])
            i += 1

    # Parse optional flags
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--x",   type=float, default=5.0)
    p.add_argument("--y",   type=float, default=0.5)
    p.add_argument("--uin", type=float, default=1.0)
    p.add_argument("--out", type=str,   default="channel_profiles.png")
    opts, _ = p.parse_known_args(opt_args)

    if len(pos_args) % 2 != 0 or len(pos_args) == 0:
        print(__doc__)
        print("ERROR: provide (pltdir label) pairs as positional arguments.")
        sys.exit(1)

    datasets = [(pos_args[i], pos_args[i+1]) for i in range(0, len(pos_args), 2)]

    # ── Figure setup ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    for idx, (pltdir, label) in enumerate(datasets):
        print(f"Reading: {pltdir}  [{label}]")
        z, u_line, time = extract_u_profile(pltdir, opts.x, opts.y)
        if np.any(np.isnan(u_line)):
            print(f"  WARNING: NaN values in profile — check FAB coverage")

        mk = _MARKERS[idx % len(_MARKERS)]
        cl = _COLORS[idx % len(_COLORS)]
        ax.plot(u_line / opts.uin, z,
                marker=mk, color=cl,
                markersize=5, markerfacecolor="none", markeredgewidth=1.0,
                linewidth=0.8, label=label)

    # Analytic Poiseuille profile for 3D square duct (first-mode approximation)
    # Not strictly correct for finite-Re with the method, but gives a reference shape.
    # Skipped here so the figure shows only simulation data.

    ax.set_xlabel(r"$U_x\,/\,U_{in}$", fontsize=13)
    ax.set_ylabel(r"$L_z$",             fontsize=13)
    ax.set_title(
        f"Fully developed velocity profile  ($x={opts.x}$, $y={opts.y}$)",
        fontsize=10,
    )
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.tick_params(direction="in", which="both", labelsize=11)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")

    fig.tight_layout()
    fig.savefig(opts.out, dpi=200, bbox_inches="tight")
    print(f"Saved: {opts.out}")


if __name__ == "__main__":
    main()
