include $(top_srcdir)/PATHS

if HAVE_MPI
if WANT_PARALLEL_MERGE
SUBDIRS = parallel
endif
endif

bin_PROGRAMS = mpi2prv

common_FILES = \
 common/fdz.h \
 common/dump.c common/dump.h \
 common/checkoptions.h common/checkoptions.c \
 common/communicators.c common/communicators.h \
 common/cpunode.c common/cpunode.h \
 common/mpi2out.c common/mpi2out.h \
 common/semantics.c common/semantics.h

dimemas_FILES = \
 dimemas/dimemas_generator.c dimemas/dimemas_generator.h \
 dimemas/mpi_trf_semantics.c dimemas/mpi_trf_semantics.h \
 dimemas/misc_trf_semantics.c dimemas/misc_trf_semantics.h \
 dimemas/trace_to_trf.c dimemas/trace_to_trf.h

paraver_FILES = \
 paraver/bswap.c paraver/bswap.h \
 paraver/trace_communication.c paraver/trace_communication.h \
 paraver/paraver_generator.c paraver/paraver_generator.h \
 paraver/paraver_state.c paraver/paraver_state.h \
 paraver/trace_to_prv.c paraver/trace_to_prv.h \
 paraver/mpi_prv_semantics.c paraver/mpi_prv_semantics.h \
 paraver/pacx_prv_semantics.c paraver/pacx_prv_semantics.h \
 paraver/omp_prv_semantics.c paraver/omp_prv_semantics.h \
 paraver/misc_prv_semantics.c paraver/misc_prv_semantics.h \
 paraver/pthread_prv_semantics.c paraver/pthread_prv_semantics.h \
 paraver/trt_prv_semantics.c paraver/trt_prv_semantics.h \
 paraver/mpi_prv_events.c paraver/mpi_prv_events.h \
 paraver/pacx_prv_events.c paraver/pacx_prv_events.h \
 paraver/omp_prv_events.c paraver/omp_prv_events.h \
 paraver/misc_prv_events.c paraver/misc_prv_events.h \
 paraver/pthread_prv_events.c paraver/pthread_prv_events.h \
 paraver/trt_prv_events.c paraver/trt_prv_events.h \
 paraver/write_file_buffer.c paraver/write_file_buffer.h \
 paraver/MPI_EventEncoding.h \
 paraver/PACX_EventEncoding.h

mpi2prv_SOURCES = \
 merger.c \
 paraver/object_tree.h \
 paraver/addr2info.c paraver/addr2info.h \
 paraver/addr2info_hashcache.c paraver/addr2info_hashcache.h \
 paraver/file_set.c paraver/file_set.h \
 paraver/communication_queues.c paraver/communication_queues.h \
 paraver/mpi_comunicadors.c paraver/mpi_comunicadors.h \
 paraver/labels.c paraver/labels.h \
 paraver/HardwareCounters.c paraver/HardwareCounters.h \
 $(dimemas_FILES) \
 $(paraver_FILES) \
 $(common_FILES) \
 $(top_srcdir)/src/common/events.h \
 $(top_srcdir)/src/common/queue.h \
 $(top_srcdir)/src/common/record.h \
 $(top_srcdir)/src/common/trace_mode.h

mpi2prv_CFLAGS = -I$(top_srcdir)/src/common -I$(top_srcdir)/src/license -I$(MERGER_DIR)/common -I$(PRV_MERGER_DIR) -I$(TRF_MERGER_DIR) @MPI_CFLAGS@ @BFD_CFLAGS@ @LIBERTY_CFLAGS@
mpi2prv_LDADD = $(COMMON_DIR)/libcommon.la
mpi2prv_LDFLAGS  = @BFD_LDFLAGS@ @BFD_LIBS@ @LIBERTY_LDFLAGS@ @LIBERTY_LIBS@
if HAVE_ZLIB
  mpi2prv_LDFLAGS += @LIBZ_LDFLAGS@ @LIBZ_LIBS@
  mpi2prv_CFLAGS += @LIBZ_CFLAGS@
endif
if HAVE_PAPI
  mpi2prv_CFLAGS += @PAPI_CFLAGS@
if IS_BG_MACHINE
  mpi2prv_LDFLAGS += -all-static @PAPI_LDFLAGS@ @PAPI_LIBS@
else
  mpi2prv_LDFLAGS += @PAPI_LDFLAGS@ -R @PAPI_SHAREDLIBSDIR@ @PAPI_LIBS@
endif
endif

if HAVE_PMAPI
  mpi2prv_LDFLAGS += -lpmapi
endif

install-exec-hook:
	cd $(DESTDIR)$(bindir) ; $(LN_S) -f mpi2prv$(EXEEXT) mpi2dim$(EXEEXT)

