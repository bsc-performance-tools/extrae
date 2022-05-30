include $(top_srcdir)/PATHS

noinst_LTLIBRARIES  = libstats_omp.la libstats_mpi.la

MPI_STATS_SOURCES = MPI/mpi_stats.c MPI/mpi_stats.h
OMP_STATS_SOURCES = OMP/omp_utils.c OMP/omp_utils.h OMP/omp_stats.c OMP/omp_stats.h

stats_INCLUDES = \
  -I$(STATS_DIR) \
  -I$(STATS_DIR)/OMP \
  -I$(STATS_DIR)/MPI \
  -I$(COMMON_INC) \
  -I$(TRACER_INC) \
  -I$(HWC_INC) \
  -I$(CLOCKS_INC) \
  -I$(WRAPPERS_INC)/API -I$(WRAPPERS_INC)/fork  \
  -I$(top_srcdir) -I$(top_srcdir)/include \
  -I$(BURST_MODE_DIR)

libstats_omp_la_CFLAGS  = $(stats_INCLUDES)
libstats_mpi_la_CFLAGS  = $(stats_INCLUDES) @MPI_CFLAGS@ -DMPI_SUPPORT

libstats_omp_la_SOURCES  = $(OMP_STATS_SOURCES)
libstats_mpi_la_SOURCES  = $(MPI_STATS_SOURCES)
