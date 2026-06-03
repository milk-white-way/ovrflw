/**
 * @file fn_init.cpp
 * @author milk-white-way (tam.thien.nguyen@ndsu.edu)
 * @brief
 * @version 0.3
 * @date 2024-06-24
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Parser.H>

#include "fn_init.H"
#include "fn_enforce_wall_bcs.H"
#include "kn_init.H"
#include "utilities.H"

using namespace amrex;
// ================================= MODULE | INITIALIZATION =================================

/**
 * @brief This function initializes the velocity components at face centers and the pressure components at cell centers.
 *
 * @param userCtx
 * @param velCont
 * @param velContPrev
 * @param velContDiff
 * @param geom
 */
void hybrid_grid_init ( MultiFab& userCtx,
                        Array<MultiFab, AMREX_SPACEDIM>& velCont,
                        Array<MultiFab, AMREX_SPACEDIM>& velContPrev,
                        MultiFab& velCart,
                        MultiFab& velCartPrev,
                        Geometry const& geom,
                        int const& Nghost,
                        Vector<int> const& phy_bc_lo,
                        Vector<int> const& phy_bc_hi,
						GpuArray<amrex::Real, AMREX_SPACEDIM> inflow_waveform,
                        Real& time,
                        Real const& dt,
                        Vector<int> const& n_cell,
                        std::string const& ic_type,
                        Vector<amrex::Real> const& ic_velocity_static,
                        amrex::Real const& ic_pressure_static,
                        std::string const& ic_velocity_x_expr,
                        std::string const& ic_velocity_y_expr,
                        std::string const& ic_velocity_z_expr,
                        std::string const& ic_pressure_expr )
{
	BL_PROFILE_VAR("hybrid_grid_init()", hybrid_grid_init);

    Box dom(geom.Domain());
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
	amrex::Print() << "INFO| dx = " << dx[0] << " dy = " << dx[1];
#if (AMREX_SPACEDIM > 2)
    amrex::Print() << " dz = " << dx[2] << "\n";
#else
    amrex::Print() << "\n";
#endif
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    int const& n0cell = n_cell[0];
    int const& n1cell = n_cell[1];
#if (AMREX_SPACEDIM > 2)
    int const& n2cell = n_cell[2];
#endif

	BL_PROFILE_VAR("init_velocity_contravariant_components()", init_velocity_contravariant_components);

    if (ic_type != "static" && ic_type != "dynamic") {
        amrex::Abort("hybrid_grid_init: unknown ic_type '" + ic_type + "'. Valid: 'static', 'dynamic'.");
    }

    amrex::Parser px_parser, py_parser, pz_parser, pp_parser;
    amrex::ParserExecutor<4> fx, fy, fz, fp;
    if (ic_type == "dynamic") {
        px_parser.define(ic_velocity_x_expr); px_parser.registerVariables({"x","y","z","t"}); fx = px_parser.compile<4>();
        py_parser.define(ic_velocity_y_expr); py_parser.registerVariables({"x","y","z","t"}); fy = py_parser.compile<4>();
        pz_parser.define(ic_velocity_z_expr); pz_parser.registerVariables({"x","y","z","t"}); fz = pz_parser.compile<4>();
        pp_parser.define(ic_pressure_expr);   pp_parser.registerVariables({"x","y","z","t"}); fp = pp_parser.compile<4>();
    }

    Real u0 = ic_velocity_static[0];
    Real v0 = ic_velocity_static[1];
#if (AMREX_SPACEDIM > 2)
    Real w0 = ic_velocity_static[2];
#endif

// Initialize velocity components at cells' face centers
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCont[0]); mfi.isValid(); ++mfi )
    {
        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
        auto const& vel_cont_x = velCont[0].array(mfi);
        auto const& vel_cont_y = velCont[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& vel_cont_z = velCont[2].array(mfi);
#endif

        auto const& vel_cont_prev_x = velContPrev[0].array(mfi);
        auto const& vel_cont_prev_y = velContPrev[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& vel_cont_prev_z = velContPrev[2].array(mfi);
#endif

        if (ic_type == "static") {
            amrex::ParallelFor(xbx,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k){
                vel_cont_x(i, j, k)      = u0;
                vel_cont_prev_x(i, j, k) = u0;
            });
            amrex::ParallelFor(ybx,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k){
                vel_cont_y(i, j, k)      = v0;
                vel_cont_prev_y(i, j, k) = v0;
            });
#if (AMREX_SPACEDIM > 2)
            amrex::ParallelFor(zbx,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k){
                vel_cont_z(i, j, k)      = w0;
                vel_cont_prev_z(i, j, k) = w0;
            });
#endif
        } else {
            amrex::ParallelFor(xbx,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k){
                amrex::Real x = prob_lo[0] + (i + Real(0.0)) * dx[0];
                amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
                amrex::Real z = prob_lo[2] + (k + Real(0.5)) * dx[2];
                vel_cont_x(i, j, k)      = (amrex::Real)fx(x, y, z, time);
                vel_cont_prev_x(i, j, k) = (amrex::Real)fx(x, y, z, time - dt);
            });
            amrex::ParallelFor(ybx,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k){
                amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                amrex::Real y = prob_lo[1] + (j + Real(0.0)) * dx[1];
                amrex::Real z = prob_lo[2] + (k + Real(0.5)) * dx[2];
                vel_cont_y(i, j, k)      = (amrex::Real)fy(x, y, z, time);
                vel_cont_prev_y(i, j, k) = (amrex::Real)fy(x, y, z, time - dt);
            });
#if (AMREX_SPACEDIM > 2)
            amrex::ParallelFor(zbx,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k){
                amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
                amrex::Real z = prob_lo[2] + (k + Real(0.0)) * dx[2];
                vel_cont_z(i, j, k)      = (amrex::Real)fz(x, y, z, time);
                vel_cont_prev_z(i, j, k) = (amrex::Real)fz(x, y, z, time - dt);
            });
#endif
        }
    }
	BL_PROFILE_VAR_STOP(init_velocity_contravariant_components);

	BL_PROFILE_VAR("init_velocity_cartesian_components()", init_velocity_cartesian_components);
    // Initialize cartesian velocity components at cell centers
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
	for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
	{
		const Box &vbx = mfi.validbox();
		auto const &ucart_init = velCart.array(mfi);
		auto const &ucart_prev = velCartPrev.array(mfi);

        if (ic_type == "static") {
            amrex::ParallelFor(vbx,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k){
                ucart_init(i, j, k, 0) = u0;
                ucart_init(i, j, k, 1) = v0;
#if (AMREX_SPACEDIM > 2)
                ucart_init(i, j, k, 2) = w0;
#endif
                ucart_prev(i, j, k, 0) = u0;
                ucart_prev(i, j, k, 1) = v0;
#if (AMREX_SPACEDIM > 2)
                ucart_prev(i, j, k, 2) = w0;
#endif
            });
        } else {
            amrex::ParallelFor(vbx,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k){
                amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
                amrex::Real z = prob_lo[2] + (k + Real(0.5)) * dx[2];
                ucart_init(i, j, k, 0) = (amrex::Real)fx(x, y, z, time);
                ucart_init(i, j, k, 1) = (amrex::Real)fy(x, y, z, time);
#if (AMREX_SPACEDIM > 2)
                ucart_init(i, j, k, 2) = (amrex::Real)fz(x, y, z, time);
#endif
                ucart_prev(i, j, k, 0) = (amrex::Real)fx(x, y, z, time - dt);
                ucart_prev(i, j, k, 1) = (amrex::Real)fy(x, y, z, time - dt);
#if (AMREX_SPACEDIM > 2)
                ucart_prev(i, j, k, 2) = (amrex::Real)fz(x, y, z, time - dt);
#endif
            });
        }
	}
	BL_PROFILE_VAR_STOP(init_velocity_cartesian_components);

	BL_PROFILE_VAR("init_fill_boundary()", init_fill_boundary);
    // Fill ghost cells
    // Periodic boundary conditions
    // -- periodic: 111
    velCart.FillBoundary(geom.periodicity());
	BL_PROFILE_VAR_STOP(init_fill_boundary);

	BL_PROFILE_VAR("init_enforce_wall_bcs()", init_enforce_wall_bcs);
    // Physical boundary conditions
    // -- wall: 131 (no-slip), -131 (slip)
    // -- inlet: 151 (constant velocity), -151 (time-dependent velocity)
    // -- outlet: 171 (constant velocity), -171 (time-dependent velocity)
    enforce_bcs_for_velCart(velCart, geom, Nghost, phy_bc_lo, phy_bc_hi, n_cell, inflow_waveform);
	BL_PROFILE_VAR_STOP(init_enforce_wall_bcs);

	BL_PROFILE_VAR("init_pressure_components()", init_pressure_components);
// Initialize pressure components at cell centers
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(userCtx); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& ctx = userCtx.array(mfi);
        if (ic_type == "static") {
            amrex::Real p0 = ic_pressure_static;
            amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                ctx(i, j, k, 0) = p0;
                ctx(i, j, k, 1) = Real(0.0);
            });
        } else {
            amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::Real x = prob_lo[0] + (i + Real(0.5)) * dx[0];
                amrex::Real y = prob_lo[1] + (j + Real(0.5)) * dx[1];
                amrex::Real z = prob_lo[2] + (k + Real(0.5)) * dx[2];
                ctx(i, j, k, 0) = (amrex::Real)fp(x, y, z, time);
                ctx(i, j, k, 1) = Real(0.0);
            });
        }
    }

    userCtx.FillBoundary(geom.periodicity());
    enforce_bcs_for_userCtx(userCtx, geom, n_cell);
	BL_PROFILE_VAR_STOP(init_pressure_components);
}

void init (MultiFab& userCtx,
           MultiFab& velCart,
           MultiFab& velCartDiff,
           Array<MultiFab, AMREX_SPACEDIM>& velContDiff,
           Geometry const& geom)
{

    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(userCtx); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& ctx = userCtx.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_userCtx(i, j, k, ctx, dx, prob_lo);
        });
    }
    userCtx.FillBoundary(geom.periodicity());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCart); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vcart = velCart.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_cartesian_velocity(i, j, k, vcart, dx, prob_lo);
        });
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velCartDiff); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& vcart_diff = velCartDiff.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_cartesian_velocity_difference(i, j, k, vcart_diff);
        });
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(velContDiff[0]); mfi.isValid(); ++mfi )
    {
        const Box& xbx = mfi.tilebox(IntVect(AMREX_D_DECL(1,0,0)));
        const Box& ybx = mfi.tilebox(IntVect(AMREX_D_DECL(0,1,0)));
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.tilebox(IntVect(AMREX_D_DECL(0,0,1)));
#endif
        auto const& xcont_diff = velContDiff[0].array(mfi);
        auto const& ycont_diff = velContDiff[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& zcont_diff = velContDiff[2].array(mfi);
#endif
        amrex::ParallelFor(xbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            xcont_diff(i, j, k) = Real(0.0);
        });
        amrex::ParallelFor(ybx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            ycont_diff(i, j, k) = Real(0.0);
        });
#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k){
            zcont_diff(i, j, k) = Real(0.0);
        });
#endif
    }
}
