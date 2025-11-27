include $(top_srcdir)/PATHS

CC = $(MPICC)
LINK = $(LIBTOOL) --tag=CC --mode=link $(CC)

noinst_LTLIBRARIES = libmpimpi2prv.la

bin_PROGRAMS = mpimpi2prv

common_FILES = \
 ../common/fdz.h \
 ../common/dump.c ../common/dump.h \
 ../common/communicators.c ../common/communicators.h \
 ../common/intercommunicators.c ../common/intercommunicators.h \
 ../common/checkoptions.h ../common/checkoptions.c \
 ../common/cpunode.c ../common/cpunode.h \
 ../common/mpi2out.c ../common/mpi2out.h \
 ../common/options.c ../common/options.h \
 ../common/object_tree.h ../common/object_tree.c \
 ../common/semantics.c ../common/semantics.h \
 ../common/addresses.c ../common/addresses.h \
 ../common/vector.c ../common/vector.h \
 ../common/stack.c ../common/stack.h \
 ../common/thread_dependencies.c ../common/thread_dependencies.h \
 ../common/address_space.c ../common/address_space.h

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
 ../paraver/omp_prv_semantics.c ../paraver/omp_prv_semantics.h \
 ../paraver/misc_prv_semantics.c ../paraver/misc_prv_semantics.h \
 ../paraver/pthread_prv_semantics.c ../paraver/pthread_prv_semantics.h \
 ../paraver/java_prv_semantics.c ../paraver/java_prv_semantics.h \
 ../paraver/cuda_prv_semantics.c ../paraver/cuda_prv_semantics.h \
 ../paraver/hip_prv_semantics.c ../paraver/hip_prv_semantics.h \
 ../paraver/opencl_prv_semantics.c ../paraver/opencl_prv_semantics.h \
 ../paraver/openshmem_prv_semantics.c ../paraver/openshmem_prv_semantics.h \
 ../paraver/openacc_prv_semantics.c ../paraver/openacc_prv_semantics.h \
 ../paraver/gaspi_prv_semantics.c ../paraver/gaspi_prv_semantics.h \
 ../paraver/mpi_prv_events.c ../paraver/mpi_prv_events.h \
 ../paraver/omp_prv_events.c ../paraver/omp_prv_events.h \
 ../paraver/pthread_prv_events.c ../paraver/pthread_prv_events.h \
 ../paraver/misc_prv_events.c ../paraver/misc_prv_events.h \
 ../paraver/java_prv_events.c ../paraver/java_prv_events.h \
 ../paraver/cuda_prv_events.c ../paraver/cuda_prv_events.h \
 ../paraver/hip_prv_events.c ../paraver/hip_prv_events.h \
 ../paraver/opencl_prv_events.c ../paraver/opencl_prv_events.h \
 ../paraver/openshmem_prv_events.c ../paraver/openshmem_prv_events.h \
 ../paraver/openacc_prv_events.c ../paraver/openacc_prv_events.h \
 ../paraver/gaspi_prv_events.c ../paraver/gaspi_prv_events.h \
 ../paraver/write_file_buffer.c ../paraver/write_file_buffer.h \
 ../paraver/paraver_nprintf.c ../paraver/paraver_nprintf.h \
 ../paraver/MPI_EventEncoding.h

parallel_FILES = \
 mpi-aux.h mpi-tags.h \
 parallel_merge_aux.h parallel_merge_aux.c \
 tree-logistics.c tree-logistics.h

libmpimpi2prv_la_SOURCES = \
 ../paraver/addr2types.c ../paraver/addr2types.h \
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

if HAVE_PAPI
if !HAVE_PAPI_EMBEDDED
  PAPI_LINKER_FLAGS = @PAPI_LDFLAGS@ @PAPI_RPATH@ @PAPI_LIBS@
endif
endif

libmpimpi2prv_la_CFLAGS = -DPARALLEL_MERGE -I$(top_srcdir)/merger -I$(top_srcdir)/src/common -I$(top_srcdir)/include -I$(MERGER_INC)/common -I$(PRV_MERGER_INC) -I$(TRF_MERGER_INC) @MPI_CFLAGS@ @LIBADDR2LINE_CFLAGS@
libmpimpi2prv_la_LDFLAGS = -o libmpimpi2prv.la $(CFLAGS) 

mpimpi2prv_CFLAGS = -DPARALLEL_MERGE -I$(top_srcdir)/merger -I$(top_srcdir)/src/common -I$(top_srcdir)/include -I$(MERGER_INC)/common -I$(PRV_MERGER_INC) -I$(TRF_MERGER_INC) @MPI_CFLAGS@ @LIBADDR2LINE_CFLAGS@
mpimpi2prv_LDFLAGS = libmpimpi2prv.la $(COMMON_LIB)/libcommon.la -o mpimpi2prv $(PAPI_LINKER_FLAGS) @MPI_LDFLAGS@ @MPI_RPATH@ @MPI_LIBS@
if HAVE_LIBADDR2LINE
mpimpi2prv_LDADD = @LIBADDR2LINE_LDADD@
endif
mpimpi2prv_DEPENDENCIES = libmpimpi2prv.la $(COMMON_LIB)/libcommon.la

if NEED_MPI_LIB_INCLUDE
  libmpimpi2prv_la_CFLAGS += @MPI_CFLAGS@
  mpimpi2prv_CFLAGS += @MPI_CFLAGS@
  mpimpi2prv_LDFLAGS += @MPI_LDFLAGS@ @MPI_LIBS@
endif

if HAVE_ZLIB
  libmpimpi2prv_la_CFLAGS += @LIBZ_CFLAGS@
  mpimpi2prv_CFLAGS += @LIBZ_CFLAGS@
  mpimpi2prv_LDFLAGS += @LIBZ_LDFLAGS@ @LIBZ_RPATH@ @LIBZ_LIBS@
endif

if HAVE_PAPI
if !HAVE_PAPI_EMBEDDED
  libmpimpi2prv_la_CFLAGS += @PAPI_CFLAGS@
  libmpimpi2prv_la_LDFLAGS += @PAPI_LDFLAGS@ @PAPI_RPATH@ @PAPI_LIBS@
  mpimpi2prv_CFLAGS += @PAPI_CFLAGS@
endif
endif

if HAVE_PMAPI
  mpimpi2prv_LDFLAGS += -lpmapi
endif

if HAVE_SIONLIB
  libmpimpi2prv_la_CFLAGS += @SIONLIB_CFLAGS@
  mpimpi2prv_CFLAGS += @SIONLIB_CFLAGS@
  mpimpi2prv_LDFLAGS += @SIONLIB_LDFLAGS@ @SIONLIB_RPATH@ -lsionmpi_64 -lsionser_64 -lsioncom_64 -lsioncom_64_lock_none
endif

# Online support
mpimpi2prv_CFLAGS += -DHAVE_ONLINE -I$(ONLINE_INC)
libmpimpi2prv_la_CFLAGS += -DHAVE_ONLINE -I$(ONLINE_INC)

install-exec-hook:
	cd $(DESTDIR)$(bindir) ; $(LN_S) -f mpimpi2prv$(EXEEXT) mpimpi2dim$(EXEEXT)

uninstall-local:
	rm $(DESTDIR)$(bindir)/mpimpi2dim$(EXEEXT)
