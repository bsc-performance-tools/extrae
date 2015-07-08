include $(top_srcdir)/PATHS

CC = $(MPICC)
LINK = $(LIBTOOL) --tag=CC --mode=link $(CC)

common_mpi_SRCS = \
  utils_mpi.c utils_mpi.h
common_mpi_INCLUDES = -I$(COMMON_DIR) -I$(COMMON_MPI_DIR) @MPI_CFLAGS@

if HAVE_SIONLIB
common_mpi_INCLUDES += @SIONLIB_CFLAGS@
endif

noinst_LTLIBRARIES = libcommon_mpi.la
libcommon_mpi_la_SOURCES = $(common_mpi_SRCS)
libcommon_mpi_la_CFLAGS  = $(common_mpi_INCLUDES)

