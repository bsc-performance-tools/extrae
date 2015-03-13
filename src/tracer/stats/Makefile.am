include $(top_srcdir)/PATHS

noinst_LTLIBRARIES  = 

if HAVE_MPI
noinst_LTLIBRARIES += libmpistats.la
endif

# MPI stats
libmpistats_la_SOURCES = \
  $(STATS_INC)/MPI/mpi_stats.c \
  $(STATS_INC)/MPI/mpi_stats.h \
  $(STATS_INC)/MPI/mpi_utils.c \
  $(STATS_INC)/MPI/mpi_utils.h

libmpistats_la_CFLAGS  = -I$(COMMON_INC) @MPI_CFLAGS@

