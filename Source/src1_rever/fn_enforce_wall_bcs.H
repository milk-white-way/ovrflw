#ifndef SUBROUTINE_ENFORCING_BCS_H_
#define SUBROUTINE_ENFORCING_BCS_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

// ============================== UTILITY | BOUNDARY CONDITIONS ==============================
/**
 * @brief The function is the second step after AMReX "FillBoundary" function fill all the interior communication cells. It enforces the wall boundary conditions on the ghost cells.
 *
 * @param velCart
 * @param geom
 * @param Nghost
 * @param phy_bc_lo
 * @param phy_bc_hi
 * @param n_cell
 * @param inflow_waveform
 */
void enforce_bcs_for_velCart (amrex::MultiFab& velCart,
                              amrex::Geometry const& geom,
                              int const& Nghost,
                              amrex::Vector<int> const& phy_bc_lo,
                              amrex::Vector<int> const& phy_bc_hi,
                              amrex::Vector<int> const& n_cell,
                              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> inflow_waveform);

void enforce_bcs_for_userCtx (amrex::MultiFab& userCtx,
                              amrex::Geometry const& geom,
                              amrex::Vector<int> const& phy_bc_lo,
                              amrex::Vector<int> const& phy_bc_hi,
                              amrex::Vector<int> const& n_cell);

void enforce_bcs_for_fluxTotal (amrex::MultiFab& flux_total,
                                amrex::Geometry const& geom,
                                amrex::Vector<int> const& n_cell);

void enforce_volumetric_flux_conservation (amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& velStar,
                                           amrex::Geometry const& geom,
                                           amrex::Vector<int> const& phy_bc_lo,
                                           amrex::Vector<int> const& phy_bc_hi,
                                           amrex::Vector<int> const& n_cell);
#endif
