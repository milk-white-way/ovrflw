#include <AMReX_MultiFabUtil.H>

#include "fn_enforce_wall_bcs.H"
#include "kn_enforce_wall_bcs.H"

using namespace amrex;

// ============================== UTILITY | BOUNDARY CONDITIONS ==============================
void enforce_bcs_for_velCart ( MultiFab& velCart,
                               Geometry const& geom,
                               int const& Nghost,
                               Vector<int> const& phy_bc_lo,
                               Vector<int> const& phy_bc_hi,
                               Vector<int> const& n_cell,
                               GpuArray<Real, AMREX_SPACEDIM> inflow_waveform )
{
    velCart.FillBoundary(geom.periodicity());
    Box dom(geom.Domain());

    // Physical boundary conditions
    // -- wall: 131 (no-slip), -131 (slip)
    // -- inlet: 151 (constant velocity), -151 (time-dependent velocity)
    // -- outlet: 171 (constant velocity), -171 (time-dependent velocity)
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
   	for (MFIter mfi(velCart); mfi.isValid(); ++mfi) {
      	const Box &vbx = mfi.growntilebox(Nghost);
      	auto const &vel_cart = velCart.array(mfi);

        int const nxcell = n_cell[0];
        int const nycell = n_cell[1];
#if (AMREX_SPACEDIM > 2)
        int const nzcell = n_cell[2];
#endif

		auto const &west_wall_bcs = phy_bc_lo[0]; 	// x- wall
		auto const &east_wall_bcs = phy_bc_hi[0]; 	// x+ wall
	 	auto const &south_wall_bcs = phy_bc_lo[1];	// y- wall
		auto const &north_wall_bcs = phy_bc_hi[1];	// y+ wall
#if (AMREX_SPACEDIM > 2)
		auto const &bakward_wall_bcs = phy_bc_lo[2]; // z- wall
		auto const &forward_wall_bcs = phy_bc_hi[2]; // z+ wall
#endif

		auto const &inflow_x = inflow_waveform[0];
		auto const &inflow_y = inflow_waveform[1];
#if (AMREX_SPACEDIM > 2)
		auto const &inflow_z = inflow_waveform[2];
#endif

        int lo = dom.smallEnd(0);
		int hi = dom.bigEnd(0);

		// Ghost cells to the left (of the West wall)
		if (vbx.smallEnd(0) < lo) {
			if (west_wall_bcs == 131) {
				amrex::ParallelFor(vbx,
                             	   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (i < lo) {
						for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = -vel_cart(-i - 1, j, k, dir);
						}
					}
				});
			} else if (west_wall_bcs == -131) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (i < lo) {
						vel_cart(i, j, k, 0) = -vel_cart(-i - 1, j, k, 0);
						for (int dir = 1; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = vel_cart(-i - 1, j, k, dir);
						}
					}
				});
			} else if (west_wall_bcs == 151) {
				amrex::ParallelFor(vbx,
							       [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (i < lo) {
					    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = 2*inflow_waveform[dir] - vel_cart(-i - 1, j, k, dir);
					    }
					}
				});
			} else if (west_wall_bcs == -151) {
				amrex::Abort("WARNING| The time-dependent inflow boundary condition is not yet implemented. Aborting! \n");
			} else if (west_wall_bcs == 171) {
			    amrex::ParallelFor(vbx,
							       [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (i < lo) {
						for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = vel_cart(-i - 1, j, k, dir);
						}
					}
				});
			} else if (west_wall_bcs == -171) {
				amrex::Abort("WARNING| The time-dependent outflow boundary condition is not yet implemented. Aborting! \n");
			}
		}

		// Ghost cells to the right (of the East wall)
		if (vbx.bigEnd(0) > hi) {
			if (east_wall_bcs == 131) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (i > hi) {
						for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = -vel_cart(((nxcell - i) + (nxcell - 1)), j, k, dir);
						}
					}
				});
			} else if (east_wall_bcs == -131) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (i > hi) {
					    vel_cart(i, j, k, 0) = -vel_cart(((nxcell - i) + (nxcell - 1)), j, k, 0);
						for (int dir = 1; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = vel_cart(((nxcell - i) + (nxcell - 1)), j, k, dir);
						}
					}
				});
			} else if (east_wall_bcs == 151) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (i > hi) {
					    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = 2*inflow_waveform[dir] - vel_cart(((nxcell - i) + (nxcell - 1)), j, k, dir);
						}
					}
				});
			} else if (east_wall_bcs == -151) {
			    amrex::Abort("WARNING| The time-dependent inflow boundary condition is not yet implemented. Aborting! \n");
			} else if (east_wall_bcs == 171) {
			    amrex::ParallelFor(vbx,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if (i > hi) {
                        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                            vel_cart(i, j, k, dir) = vel_cart(((nxcell - i) + (nxcell - 1)), j, k, dir);
                        }
                    }
				});
			} else if (east_wall_bcs == -171) {
			    amrex::Abort("WARNING| The time-dependent outflow boundary condition is not yet implemented. Aborting! \n");
			}
		}

		lo = dom.smallEnd(1);
		hi = dom.bigEnd(1);

		// Ghost cells to the bottom (of the South wall)
		if (vbx.smallEnd(1) < lo) {
			if (south_wall_bcs == 131) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (j < lo) {
						for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = -vel_cart(i, -j - 1, k, dir);
						}
					}
				});
			} else if (south_wall_bcs == -131) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (j < lo) {
						vel_cart(i, j, k, 0) = vel_cart(i, -j - 1, k, 0);
						vel_cart(i, j, k, 1) = -vel_cart(i, -j - 1, k, 1);
#if (AMREX_SPACEDIM > 2)
						vel_cart(i, j, k, 2) = vel_cart(i, -j - 1, k, 2);
#endif
					}
				});
			} else if (south_wall_bcs == 151) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (j < lo) {
						vel_cart(i, j, k, 0) = inflow_x - vel_cart(i, -j - 1, k, 0);
						vel_cart(i, j, k, 1) = inflow_y - vel_cart(i, -j - 1, k, 1);
#if (AMREX_SPACEDIM > 2)
						vel_cart(i, j, k, 2) = inflow_z - vel_cart(i, -j - 1, k, 2);
#endif
					}
				});
			} else if (south_wall_bcs == -151) {
				amrex::Abort("WARNING| The time-dependent inflow boundary condition is not yet implemented. Aborting! \n");
			} else if (south_wall_bcs == 171) {
			    amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (j < lo) {
						for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = 2*inflow_waveform[dir] - vel_cart(i, -j - 1, k, dir);
						}
					}
				});
			} else if (south_wall_bcs == -171) {
				amrex::Abort("WARNING| The time-dependent outflow boundary condition is not yet implemented. Aborting! \n");
			}
		}

		// Ghost cells to the top (of the North wall)
		if (vbx.bigEnd(1) > hi) {
			if (north_wall_bcs == 131) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (j > hi) {
						for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = -vel_cart(i, ((nycell - j) + (nycell - 1)), k, dir);
						}
					}
				});
			} else if (north_wall_bcs == -131) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (j > hi) {
						vel_cart(i, j, k, 0) = vel_cart(i, ((nycell - j) + (nycell - 1)), k, 0);
						vel_cart(i, j, k, 1) = -vel_cart(i, ((nycell - j) + (nycell - 1)), k, 1);
#if (AMREX_SPACEDIM > 2)
						vel_cart(i, j, k, 2) = vel_cart(i, ((nycell - j) + (nycell - 1)), k, 2);
#endif
					}
				});
			} else if (north_wall_bcs == 151) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (j > hi) {
							vel_cart(i, j, k, 0) = 2*inflow_x - vel_cart(i, ((nycell - j) + (nycell - 1)), k, 0);
							vel_cart(i, j, k, 1) = 2*inflow_y - vel_cart(i, ((nycell - j) + (nycell - 1)), k, 1);
#if (AMREX_SPACEDIM > 2)
							vel_cart(i, j, k, 2) = 2*inflow_z - vel_cart(i, ((nycell - j) + (nycell - 1)), k, 2);
#endif
					}
				});
			} else if (north_wall_bcs == -151) {
			    amrex::Abort("WARNING| The time-dependent inflow boundary condition is not yet implemented. Aborting! \n");
			} else if (north_wall_bcs == 171) {
			    amrex::ParallelFor(vbx,
                                 [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                  if (j > hi) {
                      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                          vel_cart(i, j, k, dir) = vel_cart(i, ((nycell - j) + (nycell - 1)), k, dir);
                      }
                  }
              });
			} else if (north_wall_bcs == -171) {
				amrex::Abort("WARNING| The time-dependent outflow boundary condition is not yet implemented. Aborting! \n");
			}
		}

#if (AMREX_SPACEDIM > 2)
		lo = dom.smallEnd(2);
		hi = dom.bigEnd(2);

		if (vbx.smallEnd(2) < lo) {
			if (bakward_wall_bcs == 131) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (k < lo) {
						for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = -vel_cart(i, j, -k -1, dir);
						}
					}
				});
			} else if (bakward_wall_bcs == -131) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (k < lo) {
					    vel_cart(i, j, k, 0) = vel_cart(i, j, -k -1, 0);
					    vel_cart(i, j, k, 1) = vel_cart(i, j, -k -1, 1);
						vel_cart(i, j, k, 2) = -vel_cart(i, j, -k -1, 2);
					}
				});
			} else if (bakward_wall_bcs == 151) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (k < lo) {
					    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                          vel_cart(i, j, k, dir) = 2*inflow_waveform[dir] - vel_cart(i, j, -k -1, dir);
                      }
					}
				});
			} else if (bakward_wall_bcs == -151) {
			    amrex::Abort("WARNING| The time-dependent inflow boundary condition is not yet implemented. Aborting! \n");
			} else if (bakward_wall_bcs == 171) {
			    amrex::ParallelFor(vbx,
                                 [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                  if (k < lo) {
                      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                          vel_cart(i, j, k, dir) = vel_cart(i, j, -k -1, dir);
                      }
                  }
              });
			} else if (bakward_wall_bcs == -195) {
			    amrex::Abort("WARNING| The time-dependent outflow boundary condition is not yet implemented. Aborting! \n");
			}
		}

		if (vbx.bigEnd(2) > hi) {
			if (forward_wall_bcs == 131) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (k > hi) {
						for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
							vel_cart(i, j, k, dir) = -vel_cart(i, j, ((nzcell - k) + (nzcell - 1)), dir);
						}
					}
				});
			} else if (forward_wall_bcs == -131) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (k > hi)	{
						vel_cart(i, j, k, 0) = vel_cart(i, j, ((nzcell - k) + (nzcell - 1)), 0);
						vel_cart(i, j, k, 1) = vel_cart(i, j, ((nzcell - k) + (nzcell - 1)), 1);
						vel_cart(i, j, k, 2) = -vel_cart(i, j, ((nzcell - k) + (nzcell - 1)), 2);
					}
				});
			} else if (forward_wall_bcs == 151) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE(int i, int j, int k) {
					if (k > hi) {
						vel_cart(i, j, k, 0) = 2 - vel_cart(i, j, ((nzcell - k) + (nzcell - 1)), 0);
						vel_cart(i, j, k, 1) = 2 - vel_cart(i, j, ((nzcell - k) + (nzcell - 1)), 1);
						vel_cart(i, j, k, 2) = 2 - vel_cart(i, j, ((nzcell - k) + (nzcell - 1)), 2);
					}
				});
			} else if (forward_wall_bcs == -151) {
				amrex::Abort("WARNING| The time-dependent inflow boundary condition is not yet implemented. Aborting! \n");
			} else if (forward_wall_bcs == 171) {
			    amrex::ParallelFor(vbx,
                                 [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                  if (k > hi)	{
                      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                          vel_cart(i, j, k, dir) = vel_cart(i, j, ((nzcell - k) + (nzcell - 1)), dir);
                      }
                  }
              });
			} else if (forward_wall_bcs == -171) {
				amrex::Abort("WARNING| The time-dependent outflow boundary condition is not yet implemented. Aborting! \n");
			}
		}
#endif
	}

}

void enforce_bcs_for_userCtx ( MultiFab& userCtx,
                               Geometry const& geom,
                               Vector<int> const& phy_bc_lo,
                               Vector<int> const& phy_bc_hi,
                               Vector<int> const& n_cell )
{
    userCtx.FillBoundary(geom.periodicity());
    Box dom(geom.Domain());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
	for (MFIter mfi(userCtx); mfi.isValid(); ++mfi)
	{
		const Box& vbx = mfi.growntilebox(1);
		auto const& ctx = userCtx.array(mfi);

		int const& nxcell = n_cell[0];
		int const& nycell = n_cell[1];
#if (AMREX_SPACEDIM > 2)
        int const& nzcell = n_cell[2];
#endif

        auto const &west_wall_bcs = phy_bc_lo[0]; 	// x- wall
		auto const &east_wall_bcs = phy_bc_hi[0]; 	// x+ wall
	 	auto const &south_wall_bcs = phy_bc_lo[1];	// y- wall
		auto const &north_wall_bcs = phy_bc_hi[1];	// y+ wall
#if (AMREX_SPACEDIM > 2)
		auto const &bakward_wall_bcs = phy_bc_lo[2]; // z- wall
		auto const &forward_wall_bcs = phy_bc_hi[2]; // z+ wall
#endif

		int lo = dom.smallEnd(0); //amrex::Print() << lo << "\n";
		int hi = dom.bigEnd(0);   //amrex::Print() << hi << "\n";

		if (vbx.smallEnd(0) < lo) {
		    if (west_wall_bcs != 171 && west_wall_bcs != -171) {
				amrex::ParallelFor(vbx,
							       [=] AMREX_GPU_DEVICE (int i, int j, int k) {
					if ( i < lo ) {
					    ctx(i, j, k, 0) = ctx(-i - 1, j, k, 0);
						ctx(i, j, k, 1) = ctx(-i - 1, j, k, 1);
					}
				});
			} else if (west_wall_bcs == 171 && west_wall_bcs == -171) {
                amrex::ParallelFor(vbx,
                                   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if ( i < lo ) {
                        ctx(i, j, k, 0) = -ctx(-i - 1, j, k, 0);
                        ctx(i, j, k, 1) = -ctx(-i - 1, j, k, 1);
                    }
                });
            }
		}

		if (vbx.bigEnd(0) > hi) {
		    if (east_wall_bcs != 171 && east_wall_bcs != -171) {
				amrex::ParallelFor(vbx,
							       [=] AMREX_GPU_DEVICE (int i, int j, int k) {
					if ( i > hi ) {
					    ctx(i, j, k, 0) = ctx(((nxcell - i) + (nxcell - 1)), j, k, 0);
						ctx(i, j, k, 1) = ctx(((nxcell - i) + (nxcell - 1)), j, k, 1);
					}
				});
            } else if (east_wall_bcs == 171 && east_wall_bcs == -171) {
                amrex::ParallelFor(vbx,
                                   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if ( i > hi ) {
                        ctx(i, j, k, 0) = -ctx(((nxcell - i) + (nxcell - 1)), j, k, 0);
                        ctx(i, j, k, 1) = -ctx(((nxcell - i) + (nxcell - 1)), j, k, 1);
                    }
                });
            }
		}

		lo = dom.smallEnd(1);
		hi = dom.bigEnd(1);

		if (vbx.smallEnd(1) < lo) {
		    if (south_wall_bcs != 171 && south_wall_bcs != -171) {
                amrex::ParallelFor(vbx,
                                   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if ( j < lo ) {
                        ctx(i, j, k, 0) = ctx(i, -j - 1, k, 0);
                        ctx(i, j, k, 1) = ctx(i, -j - 1, k, 1);
                    }
                });
            } else if (south_wall_bcs == 171 && south_wall_bcs == -171) {
                amrex::ParallelFor(vbx,
                                   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if ( j < lo ) {
                        ctx(i, j, k, 0) = -ctx(i, -j - 1, k, 0);
                        ctx(i, j, k, 1) = -ctx(i, -j - 1, k, 1);
                    }
                });
            }
		}

		if (vbx.bigEnd(1) > hi) {
		    if (north_wall_bcs != 171 && north_wall_bcs != -171) {
                amrex::ParallelFor(vbx,
                                   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if ( j > hi ) {
                        ctx(i, j, k, 0) = ctx(i, ((nycell - j) + (nycell - 1)), k, 0);
                        ctx(i, j, k, 1) = ctx(i, ((nycell - j) + (nycell - 1)), k, 1);
                    }
                });
            } else if (north_wall_bcs == 171 && north_wall_bcs == -171) {
                amrex::ParallelFor(vbx,
                                   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if ( j > hi ) {
                        ctx(i, j, k, 0) = -ctx(i, ((nycell - j) + (nycell - 1)), k, 0);
                        ctx(i, j, k, 1) = -ctx(i, ((nycell - j) + (nycell - 1)), k, 1);
                    }
                });
            }
		}

#if (AMREX_SPACEDIM > 2)
		lo = dom.smallEnd(2);
		hi = dom.bigEnd(2);

		if (vbx.smallEnd(2) < lo) {
		    if (bakward_wall_bcs != 171 && bakward_wall_bcs != -171) {
				amrex::ParallelFor(vbx,
							       [=] AMREX_GPU_DEVICE (int i, int j, int k) {
					if ( k < lo ) {
					    ctx(i, j, k, 0) = ctx(i, j, -k - 1, 0);
						ctx(i, j, k, 1) = ctx(i, j, -k - 1, 1);
					}
				});
			} else if (bakward_wall_bcs == 171 && bakward_wall_bcs == -171) {
                amrex::ParallelFor(vbx,
                                   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if ( k < lo ) {
                        ctx(i, j, k, 0) = -ctx(i, j, -k - 1, 0);
                        ctx(i, j, k, 1) = -ctx(i, j, -k - 1, 1);
                    }
                });
            }
		}

		if (vbx.bigEnd(2) > hi) {
		    if (forward_wall_bcs != 171 && forward_wall_bcs != -171) {
                amrex::ParallelFor(vbx,
                                   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if ( k > hi ) {
                        ctx(i, j, k, 0) = ctx(i, j, ((nzcell - k) + (nzcell - 1)), 0);
                        ctx(i, j, k, 1) = ctx(i, j, ((nzcell - k) + (nzcell - 1)), 1);
                    }
                });
            } else if (forward_wall_bcs == 171 && forward_wall_bcs == -171) {
				amrex::ParallelFor(vbx,
								   [=] AMREX_GPU_DEVICE (int i, int j, int k) {
					if ( k > hi ) {
						ctx(i, j, k, 0) = -ctx(i, j, ((nzcell - k) + (nzcell - 1)), 0);
						ctx(i, j, k, 1) = -ctx(i, j, ((nzcell - k) + (nzcell - 1)), 1);
					}
				});
            }
		}
#endif
	}
}

void enforce_bcs_for_fluxTotal ( MultiFab& fluxTotal,
                                 Geometry const& geom,
                                 Vector<int> const& n_cell )
{
    fluxTotal.FillBoundary(geom.periodicity());
    Box dom(geom.Domain());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(fluxTotal); mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.growntilebox(1);
        auto const& flux_total = fluxTotal.array(mfi);

        int const& nxcell = n_cell[0];
        int const& nycell = n_cell[1];
#if (AMREX_SPACEDIM > 2)
        int const& nzcell = n_cell[2];
#endif

        int lo = dom.smallEnd(0); //amrex::Print() << lo << "\n";
        int hi = dom.bigEnd(0);   //amrex::Print() << hi << "\n";

        if (vbx.smallEnd(0) < lo) {
            amrex::ParallelFor(vbx,
                               [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if ( i < lo ) {
                    flux_total(i, j, k, 0) = -flux_total(-i -1, j, k, 0);
                }
            });
        }

        if (vbx.bigEnd(0) > hi) {
            amrex::ParallelFor(vbx,
                               [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if ( i > hi ) {
                    flux_total(i, j, k, 0) = -flux_total(((nxcell - i) + (nxcell - 1)), j, k, 0);
                }
            });
        }

        lo = dom.smallEnd(1);
        hi = dom.bigEnd(1);

        if (vbx.smallEnd(1) < lo) {
            amrex::ParallelFor(vbx,
                               [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if ( j < lo ) {
                    flux_total(i, j, k, 1) = -flux_total(i, -j -1, k, 1);
                }
            });
        }

        if (vbx.bigEnd(1) > hi) {
            amrex::ParallelFor(vbx,
                               [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if ( j > hi ) {
                    flux_total(i, j, k, 1) = -flux_total(i, ((nycell - j) + (nycell - 1)), k, 1);
                }
            });
        }

#if (AMREX_SPACEDIM > 2)
        lo = dom.smallEnd(2);
        hi = dom.bigEnd(2);

        if (vbx.smallEnd(2) < lo) {
            amrex::ParallelFor(vbx,
                               [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if ( k < lo ) {
                    flux_total(i, j, k, 2) = -flux_total(i, j, -k -1, 2);
                }
            });
        }

        if (vbx.bigEnd(2) > hi) {
            amrex::ParallelFor(vbx,
                               [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if ( k > hi ) {
                    flux_total(i, j, k, 2) = -flux_total(i, j, ((nzcell - k) + (nzcell - 1)), 2);
                }
            });
        }
#endif
    }
}

void enforce_volumetric_flux_conservation ( Array<MultiFab, AMREX_SPACEDIM>& velStar,
                                            Geometry const& geom,
                                            Vector<int> const& phy_bc_lo,
                                            Vector<int> const& phy_bc_hi,
                                            Vector<int> const& n_cell )
{
    Box dom(geom.Domain());
    auto const& dx = geom.CellSizeArray();

    int const& nxcell = n_cell[0];
    int const& nycell = n_cell[1];
#if (AMREX_SPACEDIM > 2)
    int const& nzcell = n_cell[2];
#endif

    // ── x-direction: west inlet (151) → east outlet (171) ────────────────────
    if (phy_bc_lo[0] == 151 && phy_bc_hi[0] == 171)
    {
        int lo = dom.smallEnd(0);
        int hi = dom.bigEnd(0);

#if (AMREX_SPACEDIM > 2)
        Real face_area = dx[1] * dx[2];
        Real area_out  = nycell * nzcell * dx[1] * dx[2];
#else
        Real face_area = dx[1];
        Real area_out  = nycell * dx[1];
#endif

        Real FluxIn = amrex::ReduceSum(velStar[0], 0,
            [=] AMREX_GPU_DEVICE (Box const& bx, Array4<Real const> const& vs) -> Real {
                Real sum = Real(0.0);
                const auto bxlo = amrex::lbound(bx);
                const auto bxhi = amrex::ubound(bx);
                for (int kk = bxlo.z; kk <= bxhi.z; ++kk)
                for (int jj = bxlo.y; jj <= bxhi.y; ++jj)
                for (int ii = bxlo.x; ii <= bxhi.x; ++ii)
                    if (ii == lo) sum += vs(ii, jj, kk);
                return sum;
            }) * face_area;

        Real FluxOut = amrex::ReduceSum(velStar[0], 0,
            [=] AMREX_GPU_DEVICE (Box const& bx, Array4<Real const> const& vs) -> Real {
                Real sum = Real(0.0);
                const auto bxlo = amrex::lbound(bx);
                const auto bxhi = amrex::ubound(bx);
                for (int kk = bxlo.z; kk <= bxhi.z; ++kk)
                for (int jj = bxlo.y; jj <= bxhi.y; ++jj)
                for (int ii = bxlo.x; ii <= bxhi.x; ++ii)
                    if (ii == hi + 1) sum += vs(ii, jj, kk);
                return sum;
            }) * face_area;

        ParallelDescriptor::ReduceRealSum(FluxIn);
        ParallelDescriptor::ReduceRealSum(FluxOut);

        Real ratio = (FluxIn - FluxOut) / area_out;
        amrex::Print() << "  [FluxConserv] x  FluxIn=" << FluxIn
                       << "  FluxOut=" << FluxOut << "  correction=" << ratio << "\n";

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(velStar[0]); mfi.isValid(); ++mfi)
        {
            const Box& vbx = mfi.validbox();
            auto const& vs = velStar[0].array(mfi);

            if (vbx.bigEnd(0) > hi) {
                amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (i == hi + 1) vs(i, j, k) += ratio;
                });
            }
        }
    }

    // ── y-direction: south inlet (151) → north outlet (171) ──────────────────
    if (phy_bc_lo[1] == 151 && phy_bc_hi[1] == 171)
    {
        int lo = dom.smallEnd(1);
        int hi = dom.bigEnd(1);

#if (AMREX_SPACEDIM > 2)
        Real face_area = dx[0] * dx[2];
        Real area_out  = nxcell * nzcell * dx[0] * dx[2];
#else
        Real face_area = dx[0];
        Real area_out  = nxcell * dx[0];
#endif

        Real FluxIn = amrex::ReduceSum(velStar[1], 0,
            [=] AMREX_GPU_DEVICE (Box const& bx, Array4<Real const> const& vs) -> Real {
                Real sum = Real(0.0);
                const auto bxlo = amrex::lbound(bx);
                const auto bxhi = amrex::ubound(bx);
                for (int kk = bxlo.z; kk <= bxhi.z; ++kk)
                for (int jj = bxlo.y; jj <= bxhi.y; ++jj)
                for (int ii = bxlo.x; ii <= bxhi.x; ++ii)
                    if (jj == lo) sum += vs(ii, jj, kk);
                return sum;
            }) * face_area;

        Real FluxOut = amrex::ReduceSum(velStar[1], 0,
            [=] AMREX_GPU_DEVICE (Box const& bx, Array4<Real const> const& vs) -> Real {
                Real sum = Real(0.0);
                const auto bxlo = amrex::lbound(bx);
                const auto bxhi = amrex::ubound(bx);
                for (int kk = bxlo.z; kk <= bxhi.z; ++kk)
                for (int jj = bxlo.y; jj <= bxhi.y; ++jj)
                for (int ii = bxlo.x; ii <= bxhi.x; ++ii)
                    if (jj == hi + 1) sum += vs(ii, jj, kk);
                return sum;
            }) * face_area;

        ParallelDescriptor::ReduceRealSum(FluxIn);
        ParallelDescriptor::ReduceRealSum(FluxOut);

        Real ratio = (FluxIn - FluxOut) / area_out;
        amrex::Print() << "  [FluxConserv] y  FluxIn=" << FluxIn
                       << "  FluxOut=" << FluxOut << "  correction=" << ratio << "\n";

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(velStar[1]); mfi.isValid(); ++mfi)
        {
            const Box& vbx = mfi.validbox();
            auto const& vs = velStar[1].array(mfi);

            if (vbx.bigEnd(1) > hi) {
                amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (j == hi + 1) vs(i, j, k) += ratio;
                });
            }
        }
    }

#if (AMREX_SPACEDIM > 2)
    // ── z-direction: backward inlet (151) → forward outlet (171) ─────────────
    if (phy_bc_lo[2] == 151 && phy_bc_hi[2] == 171)
    {
        int lo = dom.smallEnd(2);
        int hi = dom.bigEnd(2);
        Real face_area = dx[0] * dx[1];
        Real area_out  = nxcell * nycell * dx[0] * dx[1];

        Real FluxIn = amrex::ReduceSum(velStar[2], 0,
            [=] AMREX_GPU_DEVICE (Box const& bx, Array4<Real const> const& vs) -> Real {
                Real sum = Real(0.0);
                const auto bxlo = amrex::lbound(bx);
                const auto bxhi = amrex::ubound(bx);
                for (int kk = bxlo.z; kk <= bxhi.z; ++kk)
                for (int jj = bxlo.y; jj <= bxhi.y; ++jj)
                for (int ii = bxlo.x; ii <= bxhi.x; ++ii)
                    if (kk == lo) sum += vs(ii, jj, kk);
                return sum;
            }) * face_area;

        Real FluxOut = amrex::ReduceSum(velStar[2], 0,
            [=] AMREX_GPU_DEVICE (Box const& bx, Array4<Real const> const& vs) -> Real {
                Real sum = Real(0.0);
                const auto bxlo = amrex::lbound(bx);
                const auto bxhi = amrex::ubound(bx);
                for (int kk = bxlo.z; kk <= bxhi.z; ++kk)
                for (int jj = bxlo.y; jj <= bxhi.y; ++jj)
                for (int ii = bxlo.x; ii <= bxhi.x; ++ii)
                    if (kk == hi + 1) sum += vs(ii, jj, kk);
                return sum;
            }) * face_area;

        ParallelDescriptor::ReduceRealSum(FluxIn);
        ParallelDescriptor::ReduceRealSum(FluxOut);

        Real ratio = (FluxIn - FluxOut) / area_out;
        amrex::Print() << "  [FluxConserv] z  FluxIn=" << FluxIn
                       << "  FluxOut=" << FluxOut << "  correction=" << ratio << "\n";

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(velStar[2]); mfi.isValid(); ++mfi)
        {
            const Box& vbx = mfi.validbox();
            auto const& vs = velStar[2].array(mfi);

            if (vbx.bigEnd(2) > hi) {
                amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (k == hi + 1) vs(i, j, k) += ratio;
                });
            }
        }
    }
#endif
}
