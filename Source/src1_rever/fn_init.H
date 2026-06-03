#ifndef SUBROUTINE_INITIALIZATION_H_
#define SUBROUTINE_INITIALIZATION_H_

#include "kn_init.H"
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <string>

// ================================= MODULE | INITIALIZATION =================================
void init ( amrex::MultiFab& userCtx,
            amrex::MultiFab& velCart,
            amrex::MultiFab& velCartDiff,
            amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContDiff,
            amrex::Geometry const& geom );

void hybrid_grid_init ( amrex::MultiFab& userCtx,
					    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velCont,
					    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velContPrev,
					    amrex::MultiFab& velCart,
						amrex::MultiFab& velCartPrev,
						amrex::Geometry const& geom,
						int const& Nghost,
						amrex::Vector<int> const& phy_bc_lo,
						amrex::Vector<int> const& phy_bc_hi,
						amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> inflow_waveform,
						amrex::Real& time,
						amrex::Real const& dt,
						amrex::Vector<int> const& n_cell,
						std::string const& ic_type,
						amrex::Vector<amrex::Real> const& ic_velocity_static,
						amrex::Real const& ic_pressure_static,
						std::string const& ic_velocity_x_expr,
						std::string const& ic_velocity_y_expr,
						std::string const& ic_velocity_z_expr,
						std::string const& ic_pressure_expr );
#endif
