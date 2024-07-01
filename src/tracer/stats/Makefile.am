include $(top_srcdir)/PATHS

noinst_LTLIBRARIES  = libstats.la

libstats_la_SOURCES = stats_module.c stats_module.h stats_types.h

MPI_STATS_SOURCES = MPI/mpi_stats.c MPI/mpi_stats.h
OMP_STATS_SOURCES = OMP/omp_utils.c OMP/omp_utils.h OMP/omp_stats.c OMP/omp_stats.h

stats_INCLUDES = \
  -I$(STATS_INC) \
  -I$(STATS_INC)/OMP \
  -I$(STATS_INC)/MPI \
  -I$(COMMON_INC) \
  -I$(TRACER_INC) \
  -I$(HWC_INC) \
  -I$(CLOCKS_INC) \
  -I$(WRAPPERS_INC)/API -I$(WRAPPERS_INC)/fork  \
  -I$(top_srcdir) -I$(top_srcdir)/include \
  -I$(BURST_INC)

libstats_la_CFLAGS  = $(stats_INCLUDES)


libstats_la_SOURCES += $(OMP_STATS_SOURCES)

if HAVE_MPI
libstats_la_SOURCES += $(MPI_STATS_SOURCES)
libstats_la_CFLAGS  += @MPI_CFLAGS@ -DMPI_SUPPORT
endif