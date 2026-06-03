# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../../../../amrex

USE_MPI      = TRUE
MPI_THREAD_MULTIPLE = FALSE  # amrex.async_out=1 with more than 64 processes requires MPI_THREAD_MULTIPLE
USE_OMP      = FALSE
USE_HYPRE    = FALSE
USE_CUDA     = TRUE
COMP         = gnu
DIM          = 3

DEBUG        = FALSE
TINY_PROFILE = TRUE
PROFILE      = FALSE

#FORT_CPP = cpp -E -traditional-cpp -P -x c
BL_NO_FORT   = TRUE

EBASE = ovrflw

# On NixOS, CUDA is split across Nix store packages.
# Find the merged package (has both include/ and lib/) via glob — faster than find.
CUDA_PATH := $(shell ls -d /nix/store/*-cuda-merged-* 2>/dev/null | grep -vE '(\.drv|_[a-z])' | head -1)

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

# NixOS uses lib/ not lib64/ — add it so the linker finds libcudart etc.
LIBRARY_LOCATIONS += $(CUDA_PATH)/lib

#include ../Source/src1/Make.package
#VPATH_LOCATIONS  += ../Source/src1
#INCLUDE_LOCATIONS += ../Source/src1

include ../Source/src1_rever/Make.package
VPATH_LOCATIONS  += ../Source/src1_rever
INCLUDE_LOCATIONS += ../Source/src1_rever

Pdirs 	:= Base Boundary LinearSolvers
Ppack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
