include $(top_srcdir)/PATHS

iface_MPI_SRCS     = mpi_interface.c mpi_interface.h
iface_MPI_INCLUDES =     \
   -I.                   \
   -I$(COMMON_DIR)       \
   -I$(TRACER_DIR)       \
   -I$(CLOCKS_DIR)       \
   -I$(HWC_DIR)          \
   -I$(WRAPPERS_DIR)/API \
   -I$(WRAPPERS_DIR)/MPI \
   -I$(INTERFACES_DIR)/API \
   -I@MPI_INCLUDES@ @PAPI_CFLAGS@

noinst_LTLIBRARIES    = libiface_mpif.la libiface_mpi.la
if GENERATE_LOAD_BALANCING
  noinst_LTLIBRARIES += libiface_mpif_lb.la libiface_mpi_lb.la
endif
if HAVE_DYNINST
  noinst_LTLIBRARIES += lib_dyn_iface_mpi.la
endif

if SINGLE_MPI_LIBRARY
MPITRACE_SYMBOLS= -DC_SYMBOLS -DFORTRAN_SYMBOLS
else
MPITRACE_SYMBOLS= -DC_SYMBOLS
endif

libiface_mpi_la_SOURCES = $(iface_MPI_SRCS)
libiface_mpi_la_CFLAGS  = $(iface_MPI_INCLUDES) $(MPITRACE_SYMBOLS) -DMPI_SUPPORT

libiface_mpif_la_SOURCES = $(iface_MPI_SRCS)
libiface_mpif_la_CFLAGS  = $(iface_MPI_INCLUDES) -DFORTRAN_SYMBOLS -DMPI_SUPPORT

## DynInst modules

lib_dyn_iface_mpi_la_SOURCES = $(iface_MPI_SRCS)
lib_dyn_iface_mpi_la_CFLAGS  = $(iface_MPI_INCLUDES) $(MPITRACE_SYMBOLS) -DMPI_SUPPORT -DDYNINST_MODULE -DFORTRAN_SYMBOLS -DC_SYMBOLS

## LoadBalancing Libraries

libiface_mpi_lb_la_SOURCES = $(iface_MPI_SRCS)
libiface_mpi_lb_la_CFLAGS  = $(iface_MPI_INCLUDES) $(MPITRACE_SYMBOLS) -I$(LOAD_BALANCING_HOME) -DENABLE_LOAD_BALANCING -DMPI_SUPPORT

libiface_mpif_lb_la_SOURCES = $(iface_MPI_SRCS)
libiface_mpif_lb_la_CFLAGS  = $(iface_MPI_INCLUDES) -DFORTRAN_SYMBOLS -I$(LOAD_BALANCING_HOME) -DENABLE_LOAD_BALANCING -DMPI_SUPPORT
