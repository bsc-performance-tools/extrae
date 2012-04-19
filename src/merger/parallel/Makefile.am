include $(top_srcdir)/PATHS

CC = $(MPICC)
LINK = $(LIBTOOL) --tag=CC --mode=link $(CC)

noinst_LTLIBRARIES = libmpimpi2prv.la

bin_PROGRAMS = mpimpi2prv

common_FILES = \
 ../common/fdz.h \
 ../common/dump.c ../common/dump.h \
 ../common/communicators.c ../common/communicators.h \
 ../common/checkoptions.h ../common/checkoptions.c \
 ../common/cpunode.c ../common/cpunode.h \
 ../common/mpi2out.c ../common/mpi2out.h \
 ../common/options.c ../common/options.h \
 ../common/object_tree.h ../common/object_tree.c \
 ../common/semantics.c ../common/semantics.h \
 ../common/addresses.c ../common/addresses.h \
 ../common/vector.c ../common/vector.h \
 ../common/stack.c ../common/stack.h

dimemas_FILES = \
 ../dimemas/dimemas_generator.c ../dimemas/dimemas_generator.h \
 ../dimemas/mpi_trf_semantics.c ../dimemas/mpi_trf_semantics.h \
 ../dimemas/misc_trf_semantics.c ../dimemas/misc_trf_semantics.h \
 ../dimemas/trace_to_trf.c ../dimemas/trace_to_trf.h

paraver_FILES = \
 ../paraver/bswap.c ../paraver/bswap.h \
 ../paraver/trace_communication.c ../paraver/trace_communication.h \
 ../paraver/paraver_generator.c ../paraver/paraver_generator.h \
 ../paraver/paraver_state.c ../paraver/paraver_state.h \
 ../paraver/trace_to_prv.c ../paraver/trace_to_prv.h \
 ../paraver/mpi_prv_semantics.c ../paraver/mpi_prv_semantics.h \
 ../paraver/pacx_prv_semantics.c ../paraver/pacx_prv_semantics.h \
 ../paraver/omp_prv_semantics.c ../paraver/omp_prv_semantics.h \
 ../paraver/misc_prv_semantics.c ../paraver/misc_prv_semantics.h \
 ../paraver/pthread_prv_semantics.c ../paraver/pthread_prv_semantics.h \
 ../paraver/trt_prv_semantics.c ../paraver/trt_prv_semantics.h \
 ../paraver/cuda_prv_semantics.c ../paraver/cuda_prv_semantics.h \
 ../paraver/mpi_prv_events.c ../paraver/mpi_prv_events.h \
 ../paraver/pacx_prv_events.c ../paraver/pacx_prv_events.h \
 ../paraver/omp_prv_events.c ../paraver/omp_prv_events.h \
 ../paraver/pthread_prv_events.c ../paraver/pthread_prv_events.h \
 ../paraver/misc_prv_events.c ../paraver/misc_prv_events.h \
 ../paraver/trt_prv_events.c ../paraver/trt_prv_events.h \
 ../paraver/cuda_prv_events.c ../paraver/cuda_prv_events.h \
 ../paraver/write_file_buffer.c ../paraver/write_file_buffer.h \
 ../paraver/paraver_nprintf.c ../paraver/paraver_nprintf.h \
 ../paraver/MPI_EventEncoding.h \
 ../paraver/PACX_EventEncoding.h

parallel_FILES = \
 mpi-aux.h mpi-tags.h \
 parallel_merge_aux.h parallel_merge_aux.c \
 tree-logistics.c tree-logistics.h

libmpimpi2prv_la_SOURCES = \
 ../paraver/addr2info.c ../paraver/addr2info.h \
 ../paraver/addr2info_hashcache.c ../paraver/addr2info_hashcache.h \
 ../paraver/file_set.c ../paraver/file_set.h \
 ../paraver/communication_queues.c ../paraver/communication_queues.h \
 ../paraver/mpi_comunicadors.c ../paraver/mpi_comunicadors.h \
 ../paraver/labels.c ../paraver/labels.h \
 ../paraver/HardwareCounters.c ../paraver/HardwareCounters.h \
 $(parallel_FILES) \
 $(dimemas_FILES) \
 $(paraver_FILES) \
 $(common_FILES) \
 $(top_srcdir)/src/common/events.h \
 $(top_srcdir)/src/common/queue.h \
 $(top_srcdir)/src/common/record.h

mpimpi2prv_SOURCES = \
 ../merger.c

libmpimpi2prv_la_CFLAGS = -DPARALLEL_MERGE -I.. -I$(top_srcdir)/src/common -I$(top_srcdir)/include -I$(MERGER_DIR)/common -I$(PRV_MERGER_DIR) -I$(TRF_MERGER_DIR) @MPI_CFLAGS@ @BFD_CFLAGS@ @LIBERTY_CFLAGS@
libmpimpi2prv_la_LDFLAGS = -o libmpimpi2prv.la $(CFLAGS) 

mpimpi2prv_CFLAGS = -DPARALLEL_MERGE -I.. -I$(top_srcdir)/src/common -I$(top_srcdir)/include -I$(MERGER_DIR)/common -I$(PRV_MERGER_DIR) -I$(TRF_MERGER_DIR) @MPI_CFLAGS@ @BFD_CFLAGS@ @LIBERTY_CFLAGS@
mpimpi2prv_LDFLAGS = libmpimpi2prv.la $(COMMON_DIR)/libcommon.la -o mpimpi2prv @MPIMPI2PRV_EXTRA_LIBS@ @BFD_LDFLAGS@ @BFD_LIBS@ @LIBERTY_LDFLAGS@ @LIBERTY_LIBS@ @MPI_LDFLAGS@ 
mpimpi2prv_DEPENDENCIES = libmpimpi2prv.la

if NEED_MPI_LIB_INCLUDE
  libmpimpi2prv_la_CFLAGS += @MPI_CFLAGS@
  mpimpi2prv_CFLAGS += @MPI_CFLAGS@
  mpimpi2prv_LDFLAGS += @MPI_LDFLAGS@ @MPI_LIBS@
endif

if HAVE_ZLIB
  libmpimpi2prv_la_CFLAGS += @LIBZ_CFLAGS@
  mpimpi2prv_CFLAGS += @LIBZ_CFLAGS@
  mpimpi2prv_LDFLAGS += @LIBZ_LDFLAGS@ @LIBZ_LIBS@
endif

if HAVE_PAPI
  libmpimpi2prv_la_CFLAGS += @PAPI_CFLAGS@
  mpimpi2prv_CFLAGS += @PAPI_CFLAGS@
  mpimpi2prv_LDFLAGS += @PAPI_LDFLAGS@ -R @PAPI_SHAREDLIBSDIR@ @PAPI_LIBS@
if IS_BG_MACHINE
  mpimpi2prv_LDFLAGS += -all-static
endif
if IS_CRAY_XT_MACHINE
  mpimpi2prv_LDFLAGS += -all-static
endif
endif

if HAVE_PMAPI
  mpimpi2prv_LDFLAGS += -lpmapi
endif

install-exec-hook:
	cd $(DESTDIR)$(bindir) ; $(LN_S) -f mpimpi2prv$(EXEEXT) mpimpi2dim$(EXEEXT)

