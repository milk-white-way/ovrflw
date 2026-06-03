# OvrFlw <sub>E0 (Edition One)</sub>

> Previously: AMReX-based Exascale Simulation Software for Incompressible Flows (AMRESSIF)

A GPU-accelerated, MPI-parallel incompressible Navier-Stokes solver built on [AMReX](https://github.com/AMReX-Codes/amrex).

OvrFlw implements a **fractional step (projection) method** on a **hybrid staggered/non-staggered grid** — contravariant velocities live at face centers, Cartesian velocities and pressure live at cell centers. The momentum step uses a 4-stage Runge-Kutta pseudo-time-stepping scheme; the pressure correction step uses AMReX's algebraic multigrid (MLMG) or GMRES.

---

## Numerical method at a glance

| Component | Approach |
|---|---|
| Governing equations | Incompressible Navier-Stokes |
| Time advancement | Fractional step (Kim & Moin) |
| Momentum solver | RK4 pseudo-time-stepping |
| Convective fluxes | Hybrid staggered/non-staggered |
| Pressure solver | MLMG (default) or GMRES |
| Spatial order | 2nd order |
| Parallelism | MPI + OpenMP + CUDA (optional) |

---

## Prerequisites

### AMReX

```bash
git clone https://github.com/AMReX-Codes/amrex.git
echo "export AMREX_HOME=/path/to/amrex" >> ~/.bashrc
source ~/.bashrc
```

### MPI

Any MPI implementation (OpenMPI, MPICH) on your `PATH`.

### CUDA (optional)

Required only when `USE_CUDA = TRUE` in `Exec/GNUmakefile`. OvrFlw targets `sm_86` by default — change `CUDA_ARCH` in the makefile to match your GPU. On NixOS the CUDA toolkit is auto-detected from the Nix store; no extra configuration is needed.

---

## Build

Navigate to `Exec/` and run:

```bash
# CPU + MPI (default)
make -j4

# CPU only (no MPI)
make -j4 USE_MPI=FALSE

# GPU (CUDA)
# Edit Exec/GNUmakefile: set USE_CUDA = TRUE
make -j4
```

The executable is named `main3d.gnu.TPROF.MPI[.CUDA].ex` (flags reflect build options).

---

## Running

```bash
./main3d.gnu.TPROF.MPI.ex inputs
```

With MPI:

```bash
mpirun -np 4 ./main3d.gnu.TPROF.MPI.ex inputs
```

All simulation parameters are read from the `inputs` file at runtime — no recompilation needed to change resolution, time step, boundary conditions, or initial conditions.

---

## Inputs reference

### Domain

```
lo_phy_dim = 0.0 0.0 0.0        # physical domain lower corner
hi_phy_dim = 1.0 1.0 1.0        # physical domain upper corner
n_cell     = 64 64 64           # cells in each direction
box_size = 32 32 32             # box decomposition (tune for MPI/GPU)
```

### Time stepping

```
fixed_dt = 2.0e-3               # fixed time step (preferred)
nsteps   = 10000                # number of time steps
cfl      = 0.9                  # CFL limit (used if fixed_dt not set)
```

### Fluid properties

```
ren = 1600                      # Reynolds number
vis = 6.25e-4                   # kinematic viscosity (= 1/Re)
```

### Output

```
plot_int = 100                  # plot file interval  (≤0 = off)
chk_int  = 100                  # checkpoint interval (≤0 = off)
chk_out  = 0                    # checkpoint frame to restart from (0 = fresh start)
```

Plot files (`plt*`) open directly in VisIt or ParaView. Checkpoint files (`chk*`) allow restarts.

### Boundary conditions

Each face is specified as a three-digit code for (x, y, z) components:

| Code | Meaning |
|---|---|
| `111` | Periodic |
| `131` | No-slip wall |
| `-131` | Free-slip wall |
| `151` | Inlet (constant velocity) |
| `171` | Outlet |

```
phy_bc_lo = 131 111 111         # -x: wall, y: periodic, z: periodic
phy_bc_hi = 131 111 111
inflow_waveform = 1.0 0.0 0.0   # inlet velocity (u v w), used when bc = 151
```

### Initial conditions

#### Static — uniform field

```
ic_type            = static
ic_velocity_static = 1.0 0.0 0.0   # u v w
ic_pressure_static = 0.0
```

#### Dynamic — math expressions

Values are evaluated at every grid point using the AMReX expression parser. Variables `x`, `y`, `z`, and `t` are available; `t` equals the initial time (useful for analytical solutions that evolve from a known state at t > 0).

```
ic_type            = dynamic
ic_velocity_x_expr = "sin(x)*cos(y)*cos(z)*exp(-2.0*t)"
ic_velocity_y_expr = "-cos(x)*sin(y)*cos(z)*exp(-2.0*t)"
ic_velocity_z_expr = "0"
ic_pressure_expr   = "-(1.0/16.0)*(cos(2.0*x)+cos(2.0*y))*(cos(2.0*z)+2)"
```

The previous-timestep field (`velContPrev`) is seeded at `t - dt`, so a time-accurate analytical IC is exact to second order from the very first step.

#### Init check — verify before a long run

```
stop_after_init = 1             # write pltInit* and exit immediately
```

Two plot files are always written at startup regardless of `plot_int`:
- `pltInit00000` — flow field at `t`
- `pltInitPrev00000` — flow field at `t - dt`

---

## Solver parameters

```
PSEUDO_TIMESTEPPING  = 1        # 1 = RK4 inner loop (recommended)
IterNum              = 100      # max inner iterations per time step
momentum_tolerance   = 1.0e-10 # inner-loop convergence threshold
target_resolution    = 4        # coarsest multigrid level target
```

---

## Repository layout

```
AMRESSIF/
├── Exec/
│   ├── GNUmakefile             # build configuration
│   └── inputs                  # simulation parameters
├── Source/
│   └── src1_rever/
│       ├── main.cpp            # driver; ParmParse, time loop
│       ├── fn_init.*           # initialization (IC dispatch)
│       ├── fn_flux_calc.*      # convective + viscous flux computation
│       ├── fn_rhs.*            # momentum and Poisson RHS assembly
│       ├── momentum.*          # RK4 pseudo-time-stepping
│       ├── poisson.*           # pressure Poisson solve + velocity update
│       ├── utilities.*         # contravariant ↔ Cartesian conversion
│       ├── fn_enforce_wall_bcs.* # physical BC enforcement
│       ├── GMRES_Poisson.*     # optional GMRES Poisson solver
│       └── kn_*.H              # GPU device kernel headers
├── Tests/
├── Documentations/
└── LICENSE.md
```

---

## Team

**Thien-Tam Nguyen**
Department of Civil, Construction and Environmental Engineering, North Dakota State University
[www.bettercalltam.org](https://www.bettercalltam.org)

**Dr. Trung Le**
Department of Civil, Construction and Environmental Engineering, North Dakota State University

**Dr. Andy Nonaka**
Center for Computational Sciences and Engineering (CCSE), Lawrence Berkeley National Laboratory

---

## License

[CC BY-NC-ND 4.0](LICENSE.md) — free to use and share for non-commercial purposes with attribution; derivatives require permission.
