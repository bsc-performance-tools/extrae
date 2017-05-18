include $(top_srcdir)/PATHS

iface_MPI_SRCS     = mpi_interface.c mpi_interface.h \
                     mpi_interface_p2p_c.c mpi_interface_p2p_f.c \
                     mpi_interface_coll_c.c mpi_interface_coll_f.c \
                     mpi_interface_1sided_c.c mpi_interface_1sided_f.c \
                     mpi_interface_io_c.c mpi_interface_io_f.c \
                     mpi_interface_coll_helper.c mpi_interface_coll_helper.h \
		     dlb.h

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
if SINGLE_MPI_LIBRARY
  noinst_LTLIBRARIES += libiface_mpicf.la
endif
if HAVE_DYNINST
  noinst_LTLIBRARIES += lib_dyn_iface_mpic.la lib_dyn_iface_mpif.la 
if SINGLE_MPI_LIBRARY
  noinst_LTLIBRARIES += lib_dyn_iface_mpicf.la
endif
endif

C_SYMBOLS = -DC_SYMBOLS
F_SYMBOLS = -DFORTRAN_SYMBOLS
if SINGLE_MPI_LIBRARY
CF_SYMBOLS= $(C_SYMBOLS) $(F_SYMBOLS)
endif

libiface_mpi_la_SOURCES = $(iface_MPI_SRCS)
libiface_mpi_la_CFLAGS  = $(iface_MPI_INCLUDES) $(C_SYMBOLS) -DMPI_SUPPORT

libiface_mpif_la_SOURCES = $(iface_MPI_SRCS)
libiface_mpif_la_CFLAGS  = $(iface_MPI_INCLUDES) $(F_SYMBOLS) -DMPI_SUPPORT

if SINGLE_MPI_LIBRARY
libiface_mpicf_la_SOURCES = $(iface_MPI_SRCS)
libiface_mpicf_la_CFLAGS  = $(iface_MPI_INCLUDES) $(CF_SYMBOLS) -DMPI_SUPPORT
endif

## DynInst modules

lib_dyn_iface_mpic_la_SOURCES = $(iface_MPI_SRCS)
lib_dyn_iface_mpic_la_CFLAGS  = $(iface_MPI_INCLUDES) $(C_SYMBOLS) -DMPI_SUPPORT -DDYNINST_MODULE

lib_dyn_iface_mpif_la_SOURCES = $(iface_MPI_SRCS)
lib_dyn_iface_mpif_la_CFLAGS  = $(iface_MPI_INCLUDES) $(F_SYMBOLS) -DMPI_SUPPORT -DDYNINST_MODULE 

if SINGLE_MPI_LIBRARY
lib_dyn_iface_mpicf_la_SOURCES = $(iface_MPI_SRCS)
lib_dyn_iface_mpicf_la_CFLAGS  = $(iface_MPI_INCLUDES)  $(CF_SYMBOLS) -DMPI_SUPPORT -DDYNINST_MODULE
endif
