include $(top_srcdir)/PATHS

iface_MPI_SRCS     = mpi_interface.c mpi_interface.h \
                     mpi_interface_p2p_c.c mpi_interface_p2p_f.c \
                     mpi_interface_coll_c.c mpi_interface_coll_f.c \
                     mpi_interface_1sided_c.c mpi_interface_1sided_f.c \
                     mpi_interface_io_c.c mpi_interface_io_f.c \
                     mpi_interface_coll_helper.c mpi_interface_coll_helper.h

iface_MPI_INCLUDES =     \
   -I$(INTERFACES_INC)/MPI \
   -I$(COMMON_INC)       \
   -I$(TRACER_INC)       \
   -I$(CLOCKS_INC)       \
   -I$(HWC_INC)          \
   -I$(STATS_INC)/MPI    \
   -I$(WRAPPERS_INC)/API \
   -I$(WRAPPERS_INC)/MPI \
   -I$(INTERFACES_INC)/API \
   -I$(top_srcdir)/include \
   -I@MPI_INCLUDES@ @PAPI_CFLAGS@

noinst_LTLIBRARIES    = libiface_mpif.la libiface_mpi.la
if GENERATE_LOAD_BALANCING
  noinst_LTLIBRARIES += libiface_mpif_lb.la libiface_mpi_lb.la
endif
if HAVE_DYNINST
  noinst_LTLIBRARIES += lib_dyn_iface_mpic.la lib_dyn_iface_mpif.la
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

lib_dyn_iface_mpif_la_SOURCES = $(iface_MPI_SRCS)
lib_dyn_iface_mpif_la_CFLAGS  = $(iface_MPI_INCLUDES) $(MPITRACE_SYMBOLS) -DMPI_SUPPORT -DDYNINST_MODULE -DFORTRAN_SYMBOLS

lib_dyn_iface_mpic_la_SOURCES = $(iface_MPI_SRCS)
lib_dyn_iface_mpic_la_CFLAGS  = $(iface_MPI_INCLUDES) $(MPITRACE_SYMBOLS) -DMPI_SUPPORT -DDYNINST_MODULE

## LoadBalancing Libraries

libiface_mpi_lb_la_SOURCES = $(iface_MPI_SRCS)
libiface_mpi_lb_la_CFLAGS  = $(iface_MPI_INCLUDES) $(MPITRACE_SYMBOLS) -I$(LOAD_BALANCING_HOME)/include -DENABLE_LOAD_BALANCING -DMPI_SUPPORT

libiface_mpif_lb_la_SOURCES = $(iface_MPI_SRCS)
libiface_mpif_lb_la_CFLAGS  = $(iface_MPI_INCLUDES) -DFORTRAN_SYMBOLS -I$(LOAD_BALANCING_HOME)/include -DENABLE_LOAD_BALANCING -DMPI_SUPPORT
