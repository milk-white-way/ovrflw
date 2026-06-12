# Analysis of `fd_cc.c` — Cancer Cell DPD Solver

## Overview

`fd_cc.c` implements the **Dissipative Particle Dynamics (DPD)** solver for the Cancer Cell (CC) model within the CUD-FSI fluid-structure interaction framework. The file is ~4,700 lines of C/C++ and models the mechanical behavior of a cancer cell as a two-component structure: a deformable **membrane** and a deformable **nucleus**, both represented as spring networks on triangular surface meshes.

The cell model couples to the surrounding fluid via:
- **Traction forces** computed by the IBM fluid solver and passed in through `IBMNodes`
- **DPD fluid particle forces** (random + dissipative) for thermal fluctuations

Key language features used:
- PETSc types and macros (`PetscErrorCode`, `PetscReal`, `PetscInt`, `PetscFunctionReturn`)
- C++11 standard library (`<random>`, `std::default_random_engine`, `std::normal_distribution`)
- Eigen linear algebra (`MatrixXd`, `VectorXd`) via `variables.h`

---

## PETSc Version

**Conclusion: PETSc 3.1 (patch 8)**

### Primary Evidence

| Source | Evidence |
|--------|----------|
| `fv_main.c`, line 7 | Source comment: `"This is the version written for PETSC 3.1 / (Dec 2015)"` |
| `makefile`, lines 30–32 | Commented-out include paths: `-I.../petsc-3.1-p8/include` and `-I.../petsc-3.1-p8/linux-debug/include` |

### Corroborating API Evidence

These API usages are specific to PETSc ≤ 3.1 and were deprecated or removed in later releases:

| API / Symbol | Deprecated / Removed | Locations |
|---|---|---|
| `PetscTruth` type | Renamed to `PetscBool` in PETSc 3.2 | `variables.h` lines 106, 324, 356, 362, 411; `fv_main.c` line 28; many other files |
| `petscda.h` / `DA` type | `DA` renamed to `DM`; header restructured in PETSc 3.2 | `variables.h` lines 5, 250, 254, 343 |
| `DAVecGetArray` / `DAVecRestoreArray` | Replaced by `DMDAVecGetArray` in PETSc 3.2 | `fv_les.c` (50+ calls), `fv_fsi_move.c`, others |
| `petscdmmg.h` / `DMMG` / `DMComposite` | DMMG removed in PETSc 3.4 | `variables.h` lines 8, 403, 404 |
| `PETSC_NULL` as options prefix argument | Replaced by `NULL` in PETSc 3.4+ | `fb_beam.c`, `fv_bmv.c`, `data.c`, `data_ibm.c` |
| `#if defined(PETSC_USE_LOG)` + `PetscLogEventRegister` | Present since 2.x; confirmed 3.1 API style | `fd_cc.c` lines 51–56 (commented out) |

---

## Function Reference

All 15 functions defined in `fd_cc.c` are described below.

---

### `fd_solver_cc`

```c
PetscErrorCode fd_solver_cc(FEM_SOLVER *My_Solver, FEM_OBJECT *fem_body, PetscInt ibi)
```

**Purpose:** Top-level dispatcher for the cancer cell solver. Sets `CC_Body_Type = 2` (hardcoded) and routes to either `fd_CC_Rigid_Body` (type 1) or `fd_CC_DPD_Model` (type 2). The `ibi` argument identifies which immersed body is being solved.

---

### `fd_CC_Rigid_Body`

```c
PetscErrorCode fd_CC_Rigid_Body(FEM_SOLVER *My_Solver, FEM_OBJECT *fem_body)
```

**Purpose:** Treats the cancer cell as a rigid body. Sums all traction forces from the `Traction_x/y/z` arrays over all membrane nodes to get a net force, then integrates the center-of-mass velocity and position with a forward Euler step:

```
v_new = v_old + dt * F_total / mass
x_new = x_old + dt * v_new
```

The resulting displacement is applied uniformly to every membrane node, so the entire body translates as a unit.

---

### `fd_CC_DPD_Model`

```c
PetscErrorCode fd_CC_DPD_Model(FEM_SOLVER *My_Solver, FEM_OBJECT *fem_body, PetscInt ibi)
```

**Purpose:** Main per-timestep DPD loop. Orchestrates the full force-integration cycle in this order:

1. Call `fd_CC_Membrane_Update_DPD_Node` — internal elastic forces for membrane
2. Call `fd_CC_Nucleus_Update_DPD_Node` — internal elastic forces for nucleus
3. (Optional) `Morse_Potential_CC` — cell-cell repulsion between bodies
4. Add external traction forces: `F_external[v] += Traction × stri_v[v]`
5. Compute cytoskeleton-cytosol repulsion (piecewise tangential force between membrane and nucleus nodes within a normalized distance threshold)
6. Velocity Verlet integration for membrane and nucleus: `v_new = v_n + 0.5*dt*(F_n + F_current)/mass`
7. Collision prevention between membrane triangles (currently disabled in the source)
8. Track cell stiffness: sample node k=100, compute `stiffness = |F| / |displacement|`, write to `cc_stiffness.dat`
9. Save current positions, velocities, and forces into `_n` (previous-step) arrays for next timestep

---

### `fd_CC_Membrane_Parameters`

```c
PetscErrorCode fd_CC_Membrane_Parameters(FEM_OBJECT *fem_body, FEM_SOLVER *My_Solver)
```

**Purpose:** Reads `CC_Membrane_Parameters.dat` and initializes all membrane model parameters. Computes the following non-dimensional groups from raw physical inputs:

- **WLC spring**: contour length `Lm`, persistence length `Lp_p`, maximum extension ratio `x0`
- **Power-law spring**: exponent `nm`, stiffness coefficient `kp`
- **Bending**: bending modulus `kc`, reference angle `theta0`
- **Area constraint**: local coefficient `alpha_area_loc`, global coefficient `alpha_area_g`, target area `A0t`
- **Volume constraint**: coefficient `beta_volume`, target volume `V0t`
- **Viscous / Brownian**: damping `Gamma_T`, `Gamma_C`; DPD scale `sigma_ff`
- **Morse potential**: cutoff `r_c_morse`, equilibrium `r_0_morse`, stiffness `alpha_morse`, depth `D0_morse`
- **Adhesion model**: `el0`, `el1`, spring `Ks`, `offrate`, force `FD`

---

### `fd_CC_Nucleus_Parameters`

```c
PetscErrorCode fd_CC_Nucleus_Parameters(FEM_OBJECT *fem_body, FEM_SOLVER *My_Solver)
```

**Purpose:** Identical structure to `fd_CC_Membrane_Parameters` but reads `CC_Nucleus_Parameters.dat` and populates the `_nc` (nucleus) variants of all model parameters. Nucleus has independent geometry, mass, and mechanical properties.

---

### `fd_CC_Membrane_Update_DPD_Node`

```c
PetscErrorCode fd_CC_Membrane_Update_DPD_Node(FEM_SOLVER *My_Solver, FEM_OBJECT *fem_body)
```

**Purpose:** The core per-node force accumulation and Verlet update for all membrane nodes. Computes and sums the following forces at each node:

| Force component | Method |
|---|---|
| **WLC spring** | Worm-Like Chain: `F ∝ (0.25(1−s)⁻² − 0.25 + s)` where `s = |r|/Lm` |
| **Power-law spring** | Active for `s > 0.5`: `F ∝ (1 − 3s²/4)` |
| **Bending** | Dihedral angle between adjacent triangles; `F ∝ (θ − θ₀) × kb` per vertex |
| **Local area** | Per-triangle area vs. target `A0`; gradient force on each vertex |
| **Global area** | Global area sum vs. `A0t`; uniform correction |
| **Volume** | Divergence-theorem volume vs. `V0t`; pressure-like nodal force |
| **Dissipative (viscous)** | `F_D ∝ −Γ × (v_node − v_fluid)` using FDR matrix coupling |
| **Random (Brownian)** | Gaussian noise scaled by `√(2 k_B T Γ dt)` via `std::normal_distribution` |
| **DPD fluid** | Forces from DPD fluid particles on membrane (if `DPD_Fluid_Forces == 1`) |
| **Stretch** | Point loads on extreme nodes (if stretch test is active) |
| **Collision** | Short-range repulsion between nearby triangles |

Position and velocity are updated using a velocity-scaled Verlet scheme with damping factor `lambda`:
```
x_new = x_old + dt * lambda * v
v_new = lambda * v_old + 0.5 * dt * (F_old + F_new) / mass
```

---

### `fd_CC_Nucleus_Update_DPD_Node`

```c
PetscErrorCode fd_CC_Nucleus_Update_DPD_Node(FEM_SOLVER *My_Solver, FEM_OBJECT *fem_body)
```

**Purpose:** Structurally identical to `fd_CC_Membrane_Update_DPD_Node` but operates on all nucleus nodes using the `_nc` geometry and parameter fields (`xf_nc`, `yf_nc`, `zf_nc`, `uf_nc`, etc.). The nucleus has independent mass, spring lengths, and mechanical moduli.

---

### `fd_cc_membrane_connectivity`

```c
PetscErrorCode fd_cc_membrane_connectivity(FEM_OBJECT *fem_body)
```

**Purpose:** Builds the topological connectivity data structures for the CC membrane triangular mesh. Populates:
- `Links[i]` — pairs of node indices sharing an edge (one entry per unique edge)
- `Bending_Points[i]` — four-node stencils (two shared-edge nodes + one opposite node per adjacent triangle) used for dihedral bending force computation

These structures are precomputed once and reused every timestep by `fd_CC_Membrane_Update_DPD_Node`.

---

### `fd_cc_nucleus_connectivity`

```c
PetscErrorCode fd_cc_nucleus_connectivity(FEM_OBJECT *fem_body)
```

**Purpose:** Same connectivity construction as `fd_cc_membrane_connectivity` but for the nucleus mesh, populating the `_nc` variants of `Links` and `Bending_Points`.

---

### `Reflection_Algorithm`

```c
PetscErrorCode Reflection_Algorithm(FEM_OBJECT *fem_body)
```

**Purpose:** Applies a wall-reflection boundary condition to DPD fluid particles that have crossed the solid membrane surface. For each particle near the surface:
1. Finds the closest triangle
2. Computes an area-weighted average normal at that triangle
3. Reflects the particle's position and velocity across the surface normal

This enforces a no-penetration condition for the Lagrangian DPD fluid particles.

---

### `Sorting_Points_CC_Membrane`

```c
PetscErrorCode Sorting_Points_CC_Membrane(FEM_OBJECT *fem_body)
```

**Purpose:** Sorts all membrane nodes by their x-coordinate and stores the sorted indices in `Point_Index_max` (descending) and `Point_Index_min` (ascending). Used by `Generate_Stretch_Force_CC_Membrane` to identify the nodes at the two opposite poles of the cell for mechanical stretch tests.

---

### `Generate_Stretch_Force_CC_Membrane`

```c
PetscErrorCode Generate_Stretch_Force_CC_Membrane(FEM_OBJECT *fem_body)
```

**Purpose:** Applies equal and opposite axial stretch forces to the top 2% and bottom 2% of membrane nodes (by x-coordinate rank). Force magnitude is hardcoded to **20 pN** (20×10⁻¹² N). This replicates an optical tweezer or micropipette aspiration experiment for cell stiffness characterization.

---

### `Sorting_Points_CC_Nucleus`

```c
PetscErrorCode Sorting_Points_CC_Nucleus(FEM_OBJECT *fem_body)
```

**Purpose:** x-coordinate sort for nucleus nodes, mirrors `Sorting_Points_CC_Membrane` using the `_nc` position arrays.

---

### `Generate_Stretch_Force_CC_Nuleus`

```c
PetscErrorCode Generate_Stretch_Force_CC_Nuleus(FEM_OBJECT *fem_body)
```

**Purpose:** Applies stretch forces to the nucleus (mirrors `Generate_Stretch_Force_CC_Membrane` using `_nc` fields).

> **Note:** There is a typo in the function name — `Nuleus` should be `Nucleus`. This typo appears in both the definition here and the declaration in `variables.h`.

---

### `Morse_Potential_CC`

```c
PetscErrorCode Morse_Potential_CC(FEM_SOLVER *My_Solver, FEM_OBJECT *fem_body, PetscInt ibi)
```

**Purpose:** Computes pairwise Morse potential repulsion forces between cancer cell membrane nodes and all nodes of any other body in the simulation whose `fsi_type` is 8, 9, or 10. For each interacting node pair within the cutoff radius `r_c_morse`:

```
F = D0_morse × alpha_morse × (exp(−2α(r−r0)) − exp(−α(r−r0))) × r_hat
```

Forces are added to `F_internal` of the cancer cell and subtracted from the partner body. This prevents unphysical cell-cell overlap.

---

## Key Data Structures Accessed

### `FEM_OBJECT` (membrane fields)

| Field | Type | Description |
|---|---|---|
| `xf`, `yf`, `zf` | `PetscReal *` | Current membrane node coordinates |
| `uf`, `vf`, `wf` | `PetscReal *` | Node velocities |
| `nv1`, `nv2`, `nv3` | `PetscInt *` | Triangle vertex index lists |
| `nf` | `FEM_DOF *` | Node DOF freedom/connectivity map |
| `Links` | `FEM_DOF *` | Edge connectivity (for spring forces) |
| `Bending_Points` | `FEM_DOF *` | 4-node bending stencils |
| `Ak`, `Vk` | `FEM_DOF *` | Per-element area and volume |
| `AREA`, `VOLUME` | `PetscReal` | Current total surface area and volume |
| `A0t`, `V0t` | `PetscReal` | Target area and volume |
| `F_internal`, `F_external` | `FEM_DOF *` | Internal and external nodal force arrays |
| `F_WLC_1/2`, `F_POW_1/2` | `FEM_DOF *` | WLC and power-law force components |
| `Traction_x/y/z` | `PetscReal *` | IBM traction from fluid solver |
| `stri_v` | `PetscReal *` | Nodal area weights for traction integration |
| `xf_n`, `yf_n`, `zf_n` | `PetscReal *` | Previous-step positions |
| `uf_n`, `vf_n`, `wf_n` | `PetscReal *` | Previous-step velocities |
| `F_internal_n`, `F_external_n` | `FEM_DOF *` | Previous-step forces |
| `lambda` | `PetscReal` | Velocity damping/scaling factor |
| `CC_Membrane_Mass` | `PetscReal` | Total membrane mass |
| `r_M`, `v_M`, `t_M` | `PetscReal` | Non-dimensionalization scales |

### `FEM_OBJECT` (nucleus fields — `_nc` suffix mirrors membrane)

All membrane fields have nucleus counterparts: `xf_nc`, `yf_nc`, `uf_nc`, `F_internal_nc`, `CC_Nucleus_Mass`, `D0_nc`, `kc_nc`, etc.

### `FEM_SOLVER`

| Field | Type | Description |
|---|---|---|
| `dtim` | `PetscReal` | DPD timestep (independent from fluid `dt`) |
| `time` | `PetscReal` | Current simulation time |
| `nt1`, `nt2` | `PetscInt` | Start and end timestep indices |
| `export_frequency` | `PetscInt` | Output write interval |

---

## Algorithm Notes: Main DPD Loop (`fd_CC_DPD_Model`)

```
for each timestep:
    ├── Compute membrane internal forces     [fd_CC_Membrane_Update_DPD_Node]
    │     ├── WLC + power-law spring forces  (per edge)
    │     ├── Bending forces                 (per edge pair / dihedral)
    │     ├── Local area constraint          (per triangle)
    │     ├── Global area constraint         (global sum)
    │     ├── Volume constraint              (divergence theorem)
    │     ├── Viscous + Brownian (DPD)       (per node, random Gaussian noise)
    │     └── DPD fluid coupling             (if DPD_Fluid_Forces == 1)
    │
    ├── Compute nucleus internal forces      [fd_CC_Nucleus_Update_DPD_Node]
    │
    ├── (Optional) Morse repulsion           [Morse_Potential_CC]
    │
    ├── Add IBM traction forces              (F_ext += Traction × nodal area)
    │
    ├── Cytoskeleton-cytosol repulsion       (piecewise tangential, membrane↔nucleus)
    │
    ├── Velocity Verlet integration
    │     v_new = v_n + 0.5*dt*(F_n + F_current)/mass
    │
    ├── Track stiffness                      (sample node 100, write cc_stiffness.dat)
    │
    └── Save state to _n arrays              (positions, velocities, forces)
```

The WLC model captures the nonlinear entropic elasticity of the spectrin network; area and volume constraints enforce membrane incompressibility and enclosed cytosol; bending energy resists shape changes; the Morse potential prevents non-physical cell-cell overlap.

---

## Standalone Extraction — `dpd_standalone/`

### Goal

Extract the three DPD solvers (RBC / WBC / CC) into a self-contained executable that:
- Targets **PETSc 3.21** (current release) instead of PETSc 3.1
- Distributes cell bodies across MPI ranks (Level 1 parallelism)
- Annotates inner force loops with **Kokkos** for CPU-thread / GPU portability (Level 2)
- Has zero dependency on the fluid solver (`fv_*`, `fe_*`, `IBMNodes`)

### Project Layout

```
dpd_standalone/
├── include/dpd_types.h          ← pruned variables.h (no PETSc DA/DMMG/fluid structs)
├── src/
│   ├── main.cpp                 ← PetscInitialize, MPI rank↔cell assignment, time loop
│   ├── dpd_cc.cpp               ← fd_cc.c with PETSc 3.21 API fixes
│   ├── dpd_rbc.cpp              ← fd_rbc.c with PETSc 3.21 API fixes
│   ├── dpd_wbc.cpp              ← fd_wbc.c with PETSc 3.21 API fixes
│   └── dpd_init.cpp             ← allocation + mesh loading + cell init dispatch
├── input/
│   ├── cells.dat                ← list of bodies: fsi_type + mesh filenames
│   ├── CC_Membrane_Parameters.dat
│   ├── CC_Nucleus_Parameters.dat
│   ├── RBC_Parameters.dat
│   └── WBC_Parameters.dat
└── CMakeLists.txt
```

### PETSc 3.1 → 3.21 Migration Applied

| Change | Occurrences (cc/rbc/wbc) | Transformation |
|---|---|---|
| `#include "variables.h"` | 1 / 1 / 1 | → `#include "dpd_types.h"` |
| `PetscMalloc(N*sizeof(T), &p)` | 158 / 92 / 91 | → `PetscMalloc1(N, &p)` |
| `PetscFunctionReturn(0)` | 3 / 3 / 3 | → `PetscFunctionReturn(PETSC_SUCCESS)` |
| `PETSC_NULL` | 1 / 0 / 0 | → `NULL` |
| `PetscTruth` | 0 in DPD files | Only in `variables.h` — handled in `dpd_types.h` |
| `#undef __FUNCT__` / `#define __FUNCT__` | throughout | Left as-is — no-ops in PETSc 3.21 |

### Allocation Split

The original code scatters array allocation across three files:

| Source | What it allocates |
|---|---|
| `fv_ibm_io.c:ibm_read_tecplot` | `IBMNodes` (fluid context; not needed in standalone) |
| `fe_transfer.c:fe_cast_body_from_ibm` | `xf/yf/zf`, `xf_n`, `uf/vf/wf`, `uf_n`, `nv1/nv2/nv3`, nucleus variants, DPD fluid particle arrays |
| `fe_initialization.c:fe_init_fem_body` | `normv_x/y/z`, `stri_v`, `mf`, `nver`, `ncel`, FEM shell arrays (not needed) |
| `fd_cc.c:fd_cc_membrane_connectivity` | All spring/bending topology arrays (`Links`, `Bending_Points`, `r_*`, `F_internal`, etc.) |

In the standalone, `dpd_init.cpp:dpd_alloc_fem_object()` replaces the first three; the connectivity functions in `dpd_cc.cpp` remain unchanged.

### Mesh Loading

`dpd_init.cpp:dpd_load_mesh()` reads the IBM Tecplot binary format directly:

```
Line 1:  n_v  n_elmt
Lines:   x  y  z  (one per vertex, n_v lines)
Lines:   nv1  nv2  nv3  (1-indexed, n_elmt lines → stored as 0-indexed)
```

This replaces the `ibm_read_tecplot → fe_cast_body_from_ibm → fe_transfer_body_to_solid_solver` pipeline, cutting the dependency on `IBMNodes` entirely.

### MPI Parallelization

- **Round-robin cell assignment**: cell `i` → rank `i % nranks`
- No inter-rank communication within force loops (cells are independent)
- `MPI_Allreduce` is only needed if global diagnostics (total area, mean stiffness) are collected — not wired in this version

### Kokkos Annotation (Second Pass)

The inner loops in `fd_CC_Membrane_Update_DPD_Node` (spring loop over `Ns` edges, node force accumulation loop over `lv_max` nodes) are the targets for `Kokkos::parallel_for` + `Kokkos::atomic_add`. This pass should be done after the code compiles and produces reference output matching the original coupled solver.

### Known Gaps

- `F_external` (IBM traction forces) remains zero in standalone — IBM coupling belongs to the full FSI problem
- `DPD_Fluid_Forces` flag (`lf_max` fluid particle arrays) is set to zero; the reflection algorithm and DPD fluid coupling paths are compiled but inactive
- Stretch force functions (`Sorting_Points_CC_*`, `Generate_Stretch_Force_CC_*`) are compiled but never called — they are dead code in the original as well

---

## CFS Flow Solver Reference

### `-inlet` option — `inletprofile` values

Read in `fv_main.c:863` into global `inletprofile` (default `2`). Controls the velocity profile applied at the inlet face in `fv_bcs.c:InflowFlux` / `BoundaryCondition`.

| Value | Profile | Formula / Notes |
|-------|---------|-----------------|
| `-1` | Uniform, reverse | `uin = -1.0` |
| `0` | Tabulated radial (steady) | Reads `inlet.dat` — 101 radial points × 1001 timesteps; interpolates `uinr[i][tstep]` |
| `1` | Uniform, forward | `uin = 1.0` |
| `2` | Parabolic *(default)* | `uin = 4·Umax·y·(H−y)/H²`, `H=4.1`, `Umax=1.5` — classic Poiseuille |
| `3` | Uniform flux | `uin = Flux_in` (scalar, hardcoded to `4.104e-4` in `fv_main.c:15`; can be overridden by a waveform from `inflow.dat`). In `InflowFlux`, also area-corrects: `uin *= AreaSumOut/AreaSumIn` |
| `4` | Fully-developed pipe | `uin = 2·(1 − 4r²)`, zero for `r > 0.5` |
| `5` | Tabulated, reversed | `uin = −InletInterpolation(r, user)` |
| `6` | Same as 3 | `uin = Flux_in` (no area correction) |
| `7` | Channel parabolic | `uin = 1.5·(1 − 4·yc²)` — Poiseuille for half-channel |
| `10` | Exact duct flow | 20-term Fourier series, exact fully-developed rectangular duct solution |
| `11` | Same as 0 | `uin = InletInterpolation(r, user)` |

The `cfs_mem8p00_nuc4p25` simulation uses `-inlet 3` (uniform plug flow). With `-nondimensionalize_inflow 1` and `U_ref = Re·ν/L = 2.67e-3 × 1.2e-6 / 1e-6 ≈ 3.2e-3 m/s`, the physical inlet velocity is `Flux_in × U_ref ≈ 1.3 µm/s`.

---

### `-vis` and `-rho` — physical units for the FSI coupling

Both are read in `fv_init.c:60–61` into `UserCtx`:

```c
PetscOptionsGetReal(PETSC_NULL, "-vis", &user->vis, PETSC_NULL);        // kinematic viscosity, m²/s
PetscOptionsGetReal(PETSC_NULL, "-rho", &user->rho_fluid, PETSC_NULL);  // fluid density, kg/m³
```

**Neither enters the fluid momentum solver directly.** The fluid solver is fully non-dimensional and only ever uses `user->ren` (Re) — diffusion terms are `nu = 1/Re` in `fv_rhs.c`.

Both values are passed only once, to `fe_fluid_solid_matching_scales` (`fe_transfer.c:685`):

```c
fe_fluid_solid_matching_scales(My_Solver, fem_body,
    user[0].ren, user[0].vis, user[0].rho_fluid, user[0].dt);
```

Inside that function (`fe_transfer.c:935`):

```c
fem_body->velocity_scale = Reynolds_Number * Viscosity / fem_body->body_length;
//  U_ref = Re × ν / L_ref  =  2.67e-3 × 1.2e-6 / 1e-6  =  3.2e-3 m/s
fem_body->time_scale     = fem_body->body_length / fem_body->velocity_scale;
//  T_ref = L_ref / U_ref  =  1e-6 / 3.2e-3  ≈  312 µs

My_Solver->dtim = fluid_dt × time_scale / TIMESTEP_RATIO;
//  physical DPD timestep fed to the Verlet integrator
```

`rho_fluid` is then used to dimensionalise the non-dimensional fluid traction before it enters the DPD force loop (`fe_transfer.c:621–623`):

```c
Traction_physical = Traction_nd × rho_fluid × velocity_scale²
```

Summary:

| Option | `UserCtx` field | Used in fluid solver | Used in FSI coupling |
|--------|-----------------|---------------------|----------------------|
| `-ren` | `user->ren` | Yes — `nu = 1/Re` in every diffusion term (`fv_rhs.c`) | Yes — computes `velocity_scale` |
| `-vis` | `user->vis` | **No** | Yes — `U_ref = Re·ν/L` in `fe_fluid_solid_matching_scales` |
| `-rho` | `user->rho_fluid` | **No** | Yes — traction dimensionalisation: `F = Traction_nd·ρ·U_ref²` |

---

### Outlet BC — `bcs.dat`, not `control.dat`

There is no `-outlet` command-line option. All six boundary faces are configured in **`bcs.dat`**, read at startup in `fv_init.c:557–565`:

```c
fscanf(fd1, "%i %i %i %i %i %i\n",
    &bctype[0], &bctype[1], &bctype[2],
    &bctype[3], &bctype[4], &bctype[5]);
```

The six indices map to grid faces: **imin, imax, jmin, jmax, kmin, kmax**.

`cfs_mem8p00_nuc4p25/bcs.dat` contains `1 1 1 1 5 4`:

| Index | Face | Value | Meaning |
|-------|------|-------|---------|
| [0] | imin | 1 | Solid wall |
| [1] | imax | 1 | Solid wall |
| [2] | jmin | 1 | Solid wall |
| [3] | jmax | 1 | Solid wall |
| [4] | kmin | 5 = `INLET` | Inlet (profile set by `-inlet`) |
| [5] | kmax | 4 = `OUTLET` | Outlet |

#### `bctype` values (defined in `fv_bcs.c:21–25`)

| Value | Name | Behavior |
|-------|------|----------|
| `1` | `SOLIDWALL` | No-slip: zero velocity |
| `3` | `SYMMETRIC` | Zero normal gradient, no normal flux |
| `4` | `OUTLET` | Convective outlet with flux correction (see below) |
| `5` | `INLET` | Inlet — profile controlled by `-inlet` |
| `6` | `FARFIELD` | Far-field / free-stream |
| `0` | Interface | Zero-gradient extrapolation (multi-block) |
| `8` | Characteristic outlet | Flux-conserving via `FluxIn/FluxOut` ratio scaling |
| `14` | Interface (multi-block) | Zero-gradient + inter-block flux accounting |
| `20` | Scaled outlet | Like `4` but multiplies `ucont` by ratio instead of adding a correction |
| `25` | Mixing outlet | Special case for LV geometry |

#### What `bctype = 4` (OUTLET) does (`fv_bcs.c:5682–5769`)

1. **Zero-gradient extrapolation**: copy interior velocity to ghost cell — `ubcs[k] = ucat[k-1]`
2. **Measure outgoing flux**: sum `FluxOut` over the outlet face
3. **Compute correction**: `ratio = (FluxIn − FluxOut) / AreaOut`
4. **Apply correction uniformly**: `ucont[k-1] += ratio × |zet|`

This enforces global mass conservation (`∑FluxIn = ∑FluxOut`) every timestep, which is required for the incompressible pressure Poisson solver to have a consistent RHS.

---

### Inlet flux (`FluxInSum`) vs. momentum flux — two separate quantities

`FluxInSum` is the global sum of `ucont.z` over all inlet faces — a scalar volumetric flow rate (m³/s or non-dimensional equivalent). It is used **only** for the outlet BC correction and is never seen by the momentum RHS.

For `-inlet 3` on a Cartesian grid with uniform (0,0,w) inlet:
- Each inlet face: `ucont[k=0][j][i].z = w × ΔxΔy`
- `FluxInSum = Σ ucont.z = w × A_inlet`

The momentum RHS (`fv_rhs.c`) uses the full `ucont` and `ucat` fields locally — it never reads `FluxInSum`. The connection is indirect: `FluxInSum` → outlet correction → globally divergence-free `ucont` field → Poisson RHS is consistent.

---

## CFS Hybrid Staggered Grid — Field Layout and Algorithms

### Variable layout (Cartesian, `mx × my × mz` node grid)

| Field | Stored at | PETSc DA | Components |
|-------|-----------|----------|------------|
| `Ucat` | Cell centres | `fda` (3 DOF) | `(u, v, w)` Cartesian velocity |
| `Ucont` | Face centres, staggered per component | `fda` (3 DOF) | `(.x, .y, .z)` = face-normal flux |
| `P` | Cell centres | `da` (1 DOF) | Pressure |
| `Ubcs` | Same allocation as `Ucat` | `fda` (3 DOF) | Desired Cartesian velocity **on boundary face only** |

`Ucont` staggering within a single array index `[k][j][i]`:

| Component | Physical location | Cartesian meaning |
|-----------|------------------|-------------------|
| `ucont[k][j][i].x` | i-face between cells `(k,j,i−1)` and `(k,j,i)` | `u × ΔyΔz` |
| `ucont[k][j][i].y` | j-face between cells `(k,j−1,i)` and `(k,j,i)` | `v × ΔxΔz` |
| `ucont[k][j][i].z` | k-face between cells `(k−1,j,i)` and `(k,j,i)` | `w × ΔxΔy` |

`Ubcs` is allocated with `VecDuplicate(Csi, ...)` (`fv_init.c:913`) — same full-domain size as `Ucat`. Only boundary-layer nodes are ever written or read. The developer comment says it explicitly: `// An ugly hack, waste of memory` (`variables.h:46`). ~95% of allocated entries are unused.

---

### Metric tensors — `csi`, `eta`, `zet` (`fv_metrics.c:3`)

Computed **once** at startup by `FormMetrics`. Each is a face-normal vector **scaled by face area**:

```c
// k-face metric (fv_metrics.c:157–159) — cross product of face edge vectors
zet[k][j][i].x = dydc * dzde - dzdc * dyde;
zet[k][j][i].y =-dxdc * dzde + dzdc * dxde;
zet[k][j][i].z = dxdc * dyde - dydc * dxde;
```

For Cartesian with spacing `Δx, Δy, Δz`:
```
csi[k][j][i] = (ΔyΔz,  0,    0  )   ← area-weighted normal to i-face
eta[k][j][i] = ( 0,   ΔxΔz,  0  )   ← area-weighted normal to j-face
zet[k][j][i] = ( 0,    0,   ΔxΔy)   ← area-weighted normal to k-face
```

`|zet| = ΔxΔy` = physical face area. So `ucont.z = u·zet = w × ΔxΔy` = volumetric flux through that face.

---

### `Contra2Cart` algorithm (`fv_rhs.c:25–132`)

Converts `Ucont` (face fluxes) → `Ucat` (cell-centre Cartesian velocity). Three steps:

**Step 1** — average adjacent face fluxes to cell centre (`fv_rhs.c:94–96`):
```c
q[0] = 0.5*(ucont[k][j][i-1].x + ucont[k][j][i].x);  // ξ-flux: left & right i-face
q[1] = 0.5*(ucont[k][j-1][i].y + ucont[k][j][i].y);  // η-flux: bottom & top j-face
q[2] = 0.5*(ucont[k-1][j][i].z + ucont[k][j][i].z);  // ζ-flux: front & back k-face
```
`q` is still flux (velocity × area), not velocity yet.

**Step 2** — build metric matrix at cell centre (`fv_rhs.c:82–92`):
```c
mat[0][*] = 0.5*(csi[k][j][i-1] + csi[k][j][i])   // averaged from two adjacent i-faces
mat[1][*] = 0.5*(eta[k][j-1][i] + eta[k][j][i])
mat[2][*] = 0.5*(zet[k-1][j][i] + zet[k][j][i])
```
For Cartesian, `mat` is diagonal: `diag(ΔyΔz, ΔxΔz, ΔxΔy)`. Metric arrays are precomputed and only read here.

**Step 3** — solve `mat · ucat = q` by Cramer's rule (`fv_rhs.c:99–117`):
```c
ucat[k][j][i].x = det0 / det;
ucat[k][j][i].y = det1 / det;
ucat[k][j][i].z = det2 / det;
```
For Cartesian the diagonal `mat` collapses this to scalar divisions:
```
ucat.x = q[0] / ΔyΔz = u_avg
ucat.y = q[1] / ΔxΔz = v_avg
ucat.z = q[2] / ΔxΔy = w_avg
```

---

### Field initialization at t = 0

**Fresh start** (`-rstart` absent): `FormInitialize` (`fv_init.c:15`) zeros all fields:
```c
VecSet(user->Ucont,    0.);   // fv_init.c:29
VecSet(user->Ucat,     0.);   // fv_init.c:31
VecSet(user->P,        0.);   // fv_init.c:24
VecSet(user->Bcs.Ubcs, 0.);   // fv_init.c:32
```
BCs are first applied at the start of the time loop. Optional non-zero seeding via `-init1 N` calls `SetInitialGuessToOne` (`fv_bcs.c:6495`).

**Restart** (`-rstart <step>`): `Ucont_P_Binary_Input` (`fv_main.c:1256`) loads:
- `vfield<step>_<rank>.dat` → `Ucont`
- `pfield<step>_<rank>.dat` → `P`

Then `Contra2Cart` reconstructs `Ucat` from the loaded `Ucont`. `Ubcs` is never saved or loaded — it is a transient working array recomputed each timestep.

**Primary unknowns**: only `Ucont` and `P` are saved to disk. `Ucat` is always derived.

---

### Non-zero initial conditions — `SetInitialGuessToOne` (`fv_bcs.c:6495`)

Activated by `-init1 N` in `control.dat`. Sets `Ucont` directly (not via `ucat`):

| `-init1` | `Ucont` set to | Use case |
|----------|---------------|----------|
| `1` | `ucont.z = 1e-6 × |zet|`, others zero | Near-zero uniform k-flow |
| `2` | `ucont.{x,y,z} = csi.x, eta.y, zet.z` | Unit flux aligned with grid |
| `3` | `ucont.{x,y,z} = uin × csi.z, uin × eta.z, uin × zet.z` | Profile flow in z-direction |

After setting `Ucont`, calls `Contra2Cart` to derive `Ucat`.

**`Cart2Contra` does not exist** — there is no function that converts an arbitrary Cartesian velocity field to `Ucont`. For analytical initial conditions (e.g. Taylor-Green vortex), the options are:
1. Add a branch to `SetInitialGuessToOne` that evaluates the analytical `(u,v,w)` at each face centroid and projects: `ucont[k][j][i].z = u·zet.x + v·zet.y + w·zet.z`
2. Generate `vfield00000_0.dat` externally with the projected `Ucont` and use `-rstart 0`

For Cartesian, the projection reduces to `ucont.z = w × ΔxΔy` (velocity × face area).

---

### How BCs enter the momentum solver — full pipeline

For channel flow (`bcs.dat = 1 1 1 1 5 4`, walls imin/imax/jmin/jmax, inlet kmin, outlet kmax):

**Step 0 — Default zero** (`fv_init.c:29–32`): `Ucont`, `Ucat`, `Ubcs` all zero. No-slip walls need no further action — zero is exact.

**Step 1 — Set `ucont` at boundary faces** (continuity / no-penetration):

| Face | Action |
|------|--------|
| 4 walls | `ucont.{x or y} = 0` already — no penetration enforced by default |
| kmin inlet (`bctype=5`) | `ucont[k=0][j][i].z = uin × |zet|` set in `InflowFlux` (`fv_bcs.c:320`) |
| kmax outlet (`bctype=4`) | `ucont[k=mz-2][j][i].z += ratio × |zet|` flux correction (`fv_bcs.c:5760`) |

**Step 2 — Set `ubcs`** (desired Cartesian face velocity, for momentum):

- Inlet: `ubcs[k=0] = uin × zet/|zet|` (unit normal direction, `fv_bcs.c:310–318`)
- Outlet: `ubcs[k=mz-1] = ucat[k=mz-2]` (zero-gradient copy, `fv_bcs.c:5749`)
- Walls: `ubcs = 0` — remains from `VecSet` init

**Step 3 — Ghost-cell `ucat` by linear reflection** (`GhostNodeVelocity`, `fv_bcs.c:4776–4839`):
```c
ucat[ghost] = 2 × ubcs[face] − ucat[interior]
```

| Face | Ghost cell `ucat` | Effective face velocity |
|------|------------------|------------------------|
| Wall | `−ucat[interior]` | 0 (no-slip) |
| Inlet | `2×ubcs − ucat[k=1]` | `ubcs = uin` |
| Outlet | `ucat[k=mz-2]` | zero-gradient |

**Step 4 — Momentum RHS uses ghost `ucat` via stencil branching** (`fv_rhs.c:872–1050`):
- Interior cell: standard higher-order upwind, reads `ucat[k][j][i±1]`
- Near-boundary cell (`nvert[neighbor] > 0.1` or at grid edge): one-sided stencil reads ghost `ucat[k][j][0]` or `ucat[k][j][mx-1]`

The two roles are cleanly separated: **`ucont` enforces continuity (divergence-free) at each face; ghost-cell `ucat` carries BCs into the convective and viscous stencils.**

---

## CFS Single Timestep Walkthrough

What the code does for one complete iteration `t(n+1) = t(n) + dt`, starting from initialized fields.

### Time loop (`fv_main.c:1408`)

```
for (ti = tistart; ti < tistart + tisteps; ti++) {
    while (SC loop not converged) {
        Struc_Solver()   // solid / DPD
        Flow_Solver()    // fluid
    }
    Copy_old_values_to_memory()   // Ucont_o = Ucont, Ucont_rm1 = Ucont_o
    Output if needed
}
```

`Flow_Solver` (`fv_solvers.c:622`) runs 6 substeps in order:

1. Turbulence models (skipped when `rans=0`, `les=0`)
2. `ImplicitMomentumSolver` — predicts `Ucont*`
3. `PoissonSolver_MG` or `PoissonSolver_Hypre` — solves for pressure correction `Phi`
4. `UpdatePressure` + `Projection` — enforces divergence-free
5. `Divergence` — diagnostic check
6. `ibm_interpolation_advanced` — IBM ghost-cell update (if immersed)
7. `Ucont_P_Binary_Output` — write restart files (if output step)

---

### Substep 1 — `ImplicitMomentumSolver` (`fv_implicitsolver.c:3418`)

A fractional-step predictor: advances `Ucont` in time without enforcing divergence-free. The old pressure `P` (from the previous step, zero at step 1) enters the RHS — the predicted `Ucont*` will have a non-zero divergence, corrected later by the Poisson step.

#### BC enforcement (lines 3463–3465)

Three calls fire before the solve, in order:

**`InflowFlux` (`fv_bcs.c:97`)** does two passes:

1. Write pass: for each inlet face cell computes `uin` from the inlet profile (`-inlet` option), then overwrites:
   ```c
   ubcs[k][j][i].{x,y,z} = uin × zet / |zet|   // desired Cartesian face velocity
   ucont[k][j][i].z       = uin × |zet|          // volumetric flux
   ```
2. Read pass: re-accesses `Ucont` and sums the values just written:
   ```c
   FluxIn += ucont[k][j][i].z;
   PetscGlobalSum(&FluxIn, &FluxInSum, PETSC_COMM_WORLD);
   ```

`FluxInSum` is the MPI-reduced total inlet volumetric flow rate. Because `ucont` was just forced in the write pass, `FluxInSum` always matches the prescribed profile exactly.

**`OutflowFlux` (`fv_bcs.c:889`)** is a pure read — no write:
```c
FluxOut += ucont[k][j][i].z;   // k = mz-2 for bctype[5]==4
PetscGlobalSum(&FluxOut, &FluxOutSum, PETSC_COMM_WORLD);
```
The summation line is identical in form to `InflowFlux`. The difference is there is no preceding write: it measures whatever `ucont` the momentum solver left at the outlet face from the previous pseudo-iteration. On the very first call (step 1, first pseudo-iteration), `ucont` is zero everywhere at the outlet, so `FluxOutSum = 0`.

**`FormBCS` (`fv_bcs.c:4994`)** sweeps all 6 boundary faces by `bctype`. For the outlet (`bctype=4`, lines 5682–5769), it recomputes its own `FluxOut` and `AreaSum` independently — it does **not** reuse the value from `OutflowFlux`. The internal measurement uses `ucat[k-1] · zet[k-1]` (line 5696–5698):
```c
FluxOut += ucat[k-1][j][i].x * zet[k-1].x
         + ucat[k-1][j][i].y * zet[k-1].y
         + ucat[k-1][j][i].z * zet[k-1].z;
```
`AreaSum` is computed in the same loop as the sum of face areas `|zet[k-1]|` (line 5700–5702):
```c
lArea += sqrt(zet[k-1].x² + zet[k-1].y² + zet[k-1].z²)
```
On a Cartesian grid `zet[k-1] = (0, 0, ΔxΔy)`, so each term contributes `ΔxΔy` and `AreaSum` is the total outlet area.

Both are MPI-reduced (lines 5715–5716), then `FormBCS` overwrites `user->FluxOutSum` with its own result (line 5717). The ratio is then (line 5730):
```c
ratio = (FluxInSum − FluxOutSum) / AreaSum   // units: velocity
```
The outlet `ucont` correction (lines 5760–5765) is not a pure additive — it first sets the zero-gradient base value and then adds the correction:
```c
ucont[k-1][j][i].z = (ucat[k-1] · zet[k-1]) + ratio × |zet[k-1]|
```
The first term carries the current physical outlet velocity projected onto the face normal; the second term is the uniform velocity correction that closes the global mass balance. On Cartesian this simplifies to:
```
ucont[k-1][j][i].z = ucat[k-1][j][i].z × ΔxΔy + ratio × ΔxΔy
```

At the end of `FormBCS` (line 5976), `Contra2Cart(user)` reconstructs `Ucat` from the updated `Ucont`.

#### Pseudo-time iteration (lines 3496–3754)

A `while (normdU > atol && ... || pseudot < 1)` loop, runs at least once per timestep:

**`FormFunction1` (`fv_rhs.c:3020`)** assembles the momentum residual RHS:
```
R = −Convection + Diffusion − PressureGradient
```
- Convection: central-difference in contravariant flux form using `Ucont` and metric cross-terms
- Diffusion: `(1/Re) × Laplacian(Ucont)` with `nu = 1/ren` (`fv_rhs.c:2503`)
- Pressure gradient: finite differences of `lP` projected onto each face normal via `icsi/jeta/kzet` and `iaj/jaj/kaj`
- IBM/ghost cells (`nvert > 0.1`): `RHS = 0`

**Time derivative term** (lines 3532–3542):

For 1st-order BDF1 (default, `COEF_TIME_ACCURACY = 1`):
```c
dUcont = Ucont − Ucont_o
Rhs   -= dUcont / dt
```
For 2nd-order BDF2 (`COEF_TIME_ACCURACY > 1`):
```c
dUcont = 1.5 × Ucont − 2 × Ucont_o + 0.5 × Ucont_rm1
```
On the very first timestep, `Ucont_o = Ucont_rm1 = 0`, so BDF2 reduces to `1.5 × Ucont` (1st-order accuracy on step 1 only).

**ADI direction-splitting solve** (lines 3553–3690):

For `dir = 0, 1, 2` (i, j, k directions in sequence):
- `ImplicitSolverLHSnew04`: builds sparse matrix `A` — implicit viscous operator in one direction plus pseudo-time contribution
- `KSPSolve(ksp, RB, Ucont_i)`: BiCGSTAB(l) linear solve for the flux increment in that direction
- Scatters solution into `dUcont`

After all 3 directions:
```c
Ucont = pUcont + dUcont    // line 3692
```
`pUcont` is the copy from the start of the pseudo-step. Ghost exchange distributes the new `Ucont` to neighboring MPI ranks. Then `InflowFlux` + `OutflowFlux` + `FormBCS` are called again (lines 3722–3737) to re-enforce BCs on the updated `Ucont` before the next pseudo-iteration.

Convergence written to `Converge_dU<bi>` log file per pseudo-step (line 3717).

The full sequence inside `ImplicitMomentumSolver` is:
```
InflowFlux → OutflowFlux → FormBCS        ← initial BC enforcement
while not converged:
    FormFunction1 + ADI KSP solve
    Ucont = pUcont + dUcont
    InflowFlux → OutflowFlux → FormBCS    ← re-enforcement after each pseudo-step
```
BC enforcement happens only inside this momentum solver. The Poisson solver and Projection that follow do **not** call `InflowFlux`, `OutflowFlux`, or `FormBCS`.

**Output of this substep**: `Ucont*` — momentum-updated, not yet divergence-free.

---

### Substep 2 — Poisson solver (`fv_solvers.c:798`)

Solves for pressure correction `Phi`:
```
∇·(∇φ / J) = div(Ucont*) / dt
```
RHS = divergence of `Ucont*`. Solved by `PoissonSolver_MG` (PETSc multigrid) or `PoissonSolver_Hypre` (Hypre AMG), selected by `-poisson`. Result stored in `user->Phi`.

---

### Substep 3 — `UpdatePressure` + `Projection` (`fv_solvers.c:813–815`)

**`UpdatePressure`**: `P += Phi`

**`Projection` (`fv_poisson.c:1920`)**: corrects `Ucont*` using `Phi` (accesses `lPhi`, not the already-updated `P`):

For the i-face flux at cell `(k,j,i)`:
```c
dpdc = Phi[k][j][i+1] − Phi[k][j][i]                   // normal gradient
dpde = 0.25 × (cross-terms, j-neighbors)                 // off-diagonal metric
dpdz = 0.25 × (cross-terms, k-neighbors)                 // off-diagonal metric

ucont[k][j][i].x -= (dpdc × |icsi|² + dpde × (ieta·icsi) + dpdz × (izet·icsi))
                     × iaj × dt × st / COEF_TIME_ACCURACY
```
Same structure for j and k faces. On Cartesian all cross-terms vanish and this reduces to:
```
ucont.z -= (Phi[k+1] − Phi[k]) / Δz × ΔxΔy × dt
```
which is the standard projection `u* − dt × ∂φ/∂n × face_area`. After projection, `Ucont` is divergence-free to Poisson solver tolerance.

After `Projection`, `P` is ghost-exchanged (`DAGlobalToLocal`, lines 817–818) so it is ready for `FormFunction1` in the next timestep.

**Note on `Ucat` after projection**: `Contra2Cart` is not called after `Projection`. `Ucat` retains the values from `FormBCS → Contra2Cart` at the end of the momentum step. On the next timestep, `FormFunction1` uses the projected `lUcont` and the slightly lagged `lUcat` — a one-step inconsistency that is standard for fractional-step methods.

---

### State at end of timestep

| Field | Status after `Flow_Solver` returns |
|---|---|
| `Ucont` | Divergence-free, satisfies momentum + BCs |
| `P` | `P_prev + Phi` |
| `Ucat` | From `Contra2Cart` inside `FormBCS` (momentum step), not updated after projection |
| `Ucont_o` | Set to `Ucont` by `Copy_old_values_to_memory` (end of main loop) |
| `Ucont_rm1` | Set to old `Ucont_o` (BDF2 second back-level) |
