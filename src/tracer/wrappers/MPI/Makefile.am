include $(top_srcdir)/PATHS

# Wrappers for MPI instrumentation
WRAPPERS_MPI_C_ONLY = \
 mpi_wrapper.c mpi_wrapper.h \
 mpi_wrapper_p2p_c.c mpi_wrapper_p2p_c.h \
 mpi_wrapper_coll_c.c mpi_wrapper_coll_c.h \
 mpi_wrapper_1sided_c.c mpi_wrapper_1sided_c.h \
 mpi_wrapper_io_c.c mpi_wrapper_io_c.h \
 persistent_requests.c persistent_requests.h \
 hash_table.c hash_table.h

WRAPPERS_MPI_ALL = \
 mpi_wrapper.c mpi_wrapper.h \
 mpi_wrapper_p2p_c.c mpi_wrapper_p2p_c.h \
 mpi_wrapper_coll_c.c mpi_wrapper_coll_c.h \
 mpi_wrapper_1sided_c.c mpi_wrapper_1sided_c.h \
 mpi_wrapper_io_c.c mpi_wrapper_io_c.h \
 mpi_wrapper_p2p_f.c mpi_wrapper_p2p_f.h \
 mpi_wrapper_coll_f.c mpi_wrapper_coll_f.h \
 mpi_wrapper_1sided_f.c mpi_wrapper_1sided_f.h \
 mpi_wrapper_io_f.c mpi_wrapper_io_f.h \
 persistent_requests.c persistent_requests.h \
 hash_table.c hash_table.h

if HAVE_MPI
EXTRA_MPI_CFLAGS = -I$(COMMON_MPI_DIR) -DMPI_SUPPORT @MPI_CFLAGS@ $(PARALLEL_CFLAGS_MERGE_LIB) \
 -I$(TRACER_INC) -I$(HWC_INC) -I$(CLOCKS_INC) \
 -I$(INTERFACES_INC)/API -I$(INTERFACES_INC)/MPI \
 -I$(STATS_INC)/MPI \
 -I$(WRAPPERS_INC)/API -I$(WRAPPERS_INC)/MPI \
 -I$(top_srcdir) -I$(top_srcdir)/include -I$(COMMON_INC) $(UNWIND_CFLAGS)
endif

noinst_LTLIBRARIES  = libwrap_mpi_c.la libwrap_mpi_f.la
if SINGLE_MPI_LIBRARY
 noinst_LTLIBRARIES += libwrap_mpi_cf.la
endif

libwrap_mpi_c_la_CFLAGS = -DC_SYMBOLS $(EXTRA_MPI_CFLAGS)
libwrap_mpi_c_la_SOURCES = $(WRAPPERS_MPI_C_ONLY)

libwrap_mpi_f_la_CFLAGS = -DFORTRAN_SYMBOLS $(EXTRA_MPI_CFLAGS)
libwrap_mpi_f_la_SOURCES = $(WRAPPERS_MPI_ALL)

libwrap_mpi_cf_la_CFLAGS = -DC_SYMBOLS -DFORTRAN_SYMBOLS $(EXTRA_MPI_CFLAGS)
libwrap_mpi_cf_la_SOURCES = $(WRAPPERS_MPI_ALL)



