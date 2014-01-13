include $(top_srcdir)/PATHS

lib_LTLIBRARIES  = 

if HAVE_MPI
lib_LTLIBRARIES += libmpistats.la
endif

# MPI stats
libmpistats_la_SOURCES = \
  $(STATS_DIR)/MPI/mpi_stats.c \
  $(STATS_DIR)/MPI/mpi_stats.h \
  $(STATS_DIR)/MPI/mpi_utils.c \
  $(STATS_DIR)/MPI/mpi_stats.h

libmpistats_la_CFLAGS  = -I$(COMMON_DIR)

