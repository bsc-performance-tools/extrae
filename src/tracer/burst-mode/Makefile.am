include $(top_srcdir)/PATHS

noinst_LTLIBRARIES  = libburstmode_omp.la
if HAVE_MPI
noinst_LTLIBRARIES += libburstmode_mpi.la
noinst_LTLIBRARIES += libburstmode_ompi.la
endif

BURSTMODE_SOURCES = \
  burst_mode.c \
  burst_mode.h

libburstmode_omp_la_SOURCES = $(BURSTMODE_SOURCES)

if HAVE_MPI
libburstmode_mpi_la_SOURCES = $(BURSTMODE_SOURCES)
libburstmode_ompi_la_SOURCES = $(BURSTMODE_SOURCES)
endif

BURSTMODE_INCLUDES = \
  -I$(BURST_INC)  \
  -I$(COMMON_INC) \
  -I$(TRACER_INC) \
  -I$(HWC_INC)    \
  -I$(CLOCKS_INC) \
  -I$(INTERFACES_INC)/API   \
  -I$(INTERFACES_INC)/MPI   \
  -I$(WRAPPERS_INC)/API     \
  -I$(WRAPPERS_INC)/MPI     \
  -I$(WRAPPERS_INC)/MALLOC  \
  -I$(WRAPPERS_INC)/IO      \
  -I$(WRAPPERS_INC)/SYSCALL \
  -I$(STATS_INC)     \
  -I$(STATS_INC)/OMP \
  -I$(top_srcdir)    \
  -I$(top_srcdir)/include

if HAVE_PAPI
  BURSTMODE_INCLUDES += -I@PAPI_CFLAGS@
endif

libburstmode_omp_la_CFLAGS  = $(BURSTMODE_INCLUDES) -I$(WRAPPERS_INC)/OMP_new -DNEW_OMP_SUPPORT
if HAVE_MPI
libburstmode_mpi_la_CFLAGS  = @MPI_CFLAGS@ -DMPI_SUPPORT -I@MPI_INCLUDES@ \
  $(BURSTMODE_INCLUDES) \
  -I$(STATS_INC)/MPI \
  -I$(COMMON_MPI_DIR)

libburstmode_ompi_la_CFLAGS  = $(libburstmode_mpi_la_CFLAGS) -I$(WRAPPERS_INC)/OMP_new -DNEW_OMP_SUPPORT
endif



 