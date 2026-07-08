# Channel flow benchmark — 3D, CPU+MPI, HYPRE
# Build: make -j4  (from this directory)
# Run:   cd re0p1 && mpirun -np 4 ../channel3d.gnu.TPROF.MPI.EXE inputs

AMREX_HOME ?= ../../../../amrex

USE_MPI      = TRUE
MPI_THREAD_MULTIPLE = FALSE
USE_OMP      = FALSE
USE_HYPRE    = TRUE
USE_CUDA     = FALSE
COMP         = gnu
DIM          = 3

DEBUG        = FALSE
TINY_PROFILE = TRUE
BL_NO_FORT   = TRUE

EBASE = channel

# In Nix, HYPRE is split: headers in *-hypre-*-dev, library in *-hypre-[0-9]*
# AMREX_HYPRE_HOME controls include path (via Make.hypre); lib path added below.
HYPRE_NIX_DEV := $(shell ls -d /nix/store/*-hypre-*-dev 2>/dev/null | head -1)
HYPRE_NIX_LIB := $(shell ls -d /nix/store/*-hypre-[0-9]* 2>/dev/null | grep -v '\-dev' | head -1)
ifeq ($(HYPRE_NIX_DEV),)
  $(error HYPRE dev package not found in Nix store. Set AMREX_HYPRE_HOME manually.)
endif
AMREX_HYPRE_HOME := $(HYPRE_NIX_DEV)

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

# Add the actual libHYPRE.so path (Nix separates headers from libraries)
LIBRARY_LOCATIONS += $(HYPRE_NIX_LIB)/lib
LIBRARIES         += -Wl,-rpath,$(HYPRE_NIX_LIB)/lib

include ../../Source/src1_rever/Make.package
VPATH_LOCATIONS   += ../../Source/src1_rever
INCLUDE_LOCATIONS += ../../Source/src1_rever

Pdirs := Base Boundary LinearSolvers
Ppack += $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
