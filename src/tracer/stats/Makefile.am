include $(top_srcdir)/PATHS

noinst_LTLIBRARIES  = 

if HAVE_MPI
noinst_LTLIBRARIES += libmpistats.la
endif

# MPI stats
libmpistats_la_SOURCES = \
  MPI/mpi_stats.c \
  MPI/mpi_stats.h \
  MPI/mpi_utils.c \
  MPI/mpi_utils.h

libmpistats_la_CFLAGS  = -I$(COMMON_INC) -I$(STATS_INC) @MPI_CFLAGS@

