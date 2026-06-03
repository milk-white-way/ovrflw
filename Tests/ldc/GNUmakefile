AMREX_HOME ?= ../../../amrex

USE_MPI   = TRUE
USE_OMP   = FALSE
USE_CUDA  = FALSE
COMP      = gnu
DIM       = 2

DEBUG        = FALSE
TINY_PROFILE = TRUE
BL_NO_FORT   = TRUE

EBASE = ldc

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ../../Source/src1_rever/Make.package
VPATH_LOCATIONS   += ../../Source/src1_rever
INCLUDE_LOCATIONS += ../../Source/src1_rever

Pdirs := Base Boundary LinearSolvers
Ppack += $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
