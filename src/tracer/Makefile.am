include $(top_srcdir)/PATHS

SUBDIRS = clocks hwc interfaces stats

# Include this files/directories for the package generation
EXTRA_DIST = external \
 wrappers/OMP/intel-kmpc-11-intermediate-switch.c \
 wrappers/OMP/intel-kmpc-11-intermediate.c

if HAVE_ONLINE
  SUBDIRS += online
endif

if IS_CELL_MACHINE
  SUBDIRS += spu
endif

if ! OS_ANDROID
	LIBPTHREAD = -lpthread
endif

#
# Libraries ending with c/f are for C/Fortran libraries. If not specified, 
# they should contain symbols for both languages.
#

lib_LTLIBRARIES = libseqtrace.la 
#lib_LTLIBRARIES = libseqtrace.la libextrae-core.la 
#if HAVE_MPI
#  lib_LTLIBRARIES += libextrae-mpi.la
#endif
#if WANT_OPENMP
#  lib_LTLIBRARIES += libextrae-openmp.la
#endif

if WANT_TRT
  lib_LTLIBRARIES += libtrttrace.la
endif

if WANT_CUDAorCUPTI
  lib_LTLIBRARIES += libcudatrace.la
endif

if WANT_OPENCL
  lib_LTLIBRARIES += libocltrace.la
endif

if HAVE_PACX
  lib_LTLIBRARIES += libpacxtrace.la libpacxtracef.la
endif

if HAVE_MPI
  lib_LTLIBRARIES += libmpitrace.la libmpitracef.la
if WANT_OPENMP
  lib_LTLIBRARIES += libompitrace.la libompitracef.la
endif
if WANT_CUDAorCUPTI
  lib_LTLIBRARIES += libcudampitrace.la libcudampitracef.la
endif
if WANT_SMPSS
  lib_LTLIBRARIES += libsmpssmpitrace.la libsmpssmpitracef.la
endif
if WANT_OPENCL
  lib_LTLIBRARIES += liboclmpitrace.la liboclmpitracef.la
endif
if WANT_NANOS
endif
endif

if HAVE_MPI
if WANT_OPENMP
if WANT_CUDAorCUPTI
  lib_LTLIBRARIES += libcudaompitrace.la libcudaompitracef.la
endif
endif
endif


if WANT_UPC
  lib_LTLIBRARIES += libupctrace.la
endif

if WANT_OPENMP
  lib_LTLIBRARIES += libomptrace.la
endif
if WANT_SMPSS
  lib_LTLIBRARIES += libsmpsstrace.la
endif
if WANT_NANOS
  lib_LTLIBRARIES += libnanostrace.la
if HAVE_MPI
    lib_LTLIBRARIES += libnanosmpitrace.la libnanosmpitracef.la
endif
endif
if WANT_PTHREAD
  lib_LTLIBRARIES += libpttrace.la
if HAVE_MPI
  lib_LTLIBRARIES += libptmpitrace.la libptmpitracef.la
endif
endif

if WANT_OPENSHMEM
  lib_LTLIBRARIES += libopenshmemtrace.la
endif

if GENERATE_LOAD_BALANCING
if HAVE_MPI
  lib_LTLIBRARIES += libmpitrace-lb.la libmpitracef-lb.la
if WANT_OPENMP
  lib_LTLIBRARIES += libompitrace-lb.la libompitracef-lb.la
endif
if WANT_SMPSS
  lib_LTLIBRARIES += libsmpssmpitrace-lb.la libsmpssmpitracef-lb.la
endif
if WANT_NANOS
  lib_LTLIBRARIES += libnanosmpitrace-lb.la libnanosmpitracef-lb.la
endif
endif
endif

if HAVE_DYNINST
if HAVE_MPI
  lib_LTLIBRARIES += lib_dyn_mpitracec.la lib_dyn_mpitracef.la
if WANT_OPENMP
  lib_LTLIBRARIES += lib_dyn_ompitracec.la lib_dyn_ompitracef.la
endif
if WANT_CUDAorCUPTI
  lib_LTLIBRARIES += lib_dyn_cudampitracec.la lib_dyn_cudampitracef.la
endif
if WANT_OPENCL
  lib_LTLIBRARIES += lib_dyn_oclmpitracec.la lib_dyn_oclmpitracef.la
endif
endif
endif

WRAPPERS_CORE = \
 wrappers/API/buffers.c wrappers/API/buffers.h \
 wrappers/API/wrapper.c wrappers/API/wrapper.h \
 wrappers/API/misc_wrapper.c wrappers/API/misc_wrapper.h \
 wrappers/API/threadinfo.c wrappers/API/threadinfo.h \
 wrappers/API/trace_buffers.h \
 trace_macros.h trace_macros_mpi.h trace_macros_omp.h trace_macros_pacx.h \
 trace_hwc.h \
 wrappers/fork/fork_wrapper.c wrappers/fork/fork_wrapper.h \
 probes/fork/fork_probe.c probes/fork/fork_probe.h \
 wrappers/MALLOC/malloc_wrapper.c wrappers/MALLOC/malloc_wrapper.h \
 probes/MALLOC/malloc_probe.c probes/MALLOC/malloc_probe.h \
 wrappers/IO/io_wrapper.c wrappers/IO/io_wrapper.h \
 probes/IO/io_probe.c probes/IO/io_probe.h 

# Wrappers for CUDA instrumentation
WRAPPERS_CUDA = probes/CUDA/cuda_probe.c probes/CUDA/cuda_probe.h \
 wrappers/CUDA/cuda_common.c wrappers/CUDA/cuda_common.h
if WANT_CUDA
WRAPPERS_CUDA += wrappers/CUDA/cuda_wrapper.c wrappers/CUDA/cuda_wrapper.h
WRAPPERS_CUDA_CFLAGS = -I$(CUDA_HOME)/include
endif
if WANT_CUPTI
WRAPPERS_CUDA += wrappers/CUDA/cuda_wrapper_cupti.c wrappers/CUDA/cuda_wrapper_cupti.h
WRAPPERS_CUDA_CFLAGS = -I$(CUPTI_HOME)/include -I$(CUDA_HOME)/include
endif

WRAPPERS_OCL = probes/OPENCL/opencl_probe.c probes/OPENCL/opencl_probe.h \
 wrappers/OPENCL/opencl_wrapper.c wrappers/OPENCL/opencl_wrapper.h \
 wrappers/OPENCL/opencl_common.c wrappers/OPENCL/opencl_common.h
WRAPPERS_OCL_CFLAGS = -I$(OPENCL_HOME)/include

# Wrappers for MPI instrumentation
WRAPPERS_MPI = mpif.h wrappers/MPI/mpi_wrapper.c wrappers/MPI/mpi_wrapper.h

# Wrappers for PACX instrumentation
WRAPPERS_PACX = wrappers/PACX/pacx_wrapper.c wrappers/PACX/pacx_wrapper.h

# Probes for OpenMP instrumentation
WRAPPERS_OMP = probes/OMP/omp_probe.c probes/OMP/omp_probe.h \
 wrappers/OMP/omp-common.c wrappers/OMP/omp-common.h 
EXTRA_OMP_CFLAGS =

if WANT_OPENMP_OMPT
WRAPPERS_OMP += wrappers/OMP/ompt-wrapper.c wrappers/OMP/ompt-helper-defs.h wrappers/OMP/ompt-helper.h wrappers/OMP/ompt-helper.c 
EXTRA_OMP_CFLAGS += -DOMPT_OPENMP

else

# These files are needed by intel/ibm/gnu openmp instrumentation, but not by OMPT
WRAPPERS_OMP += wrappers/OMP/omp_wrapper.c wrappers/OMP/omp_wrapper.h

if WANT_OPENMP_INTEL
WRAPPERS_OMP += wrappers/OMP/intel-kmpc-11.c wrappers/OMP/intel-kmpc-11.h wrappers/OMP/intel-kmpc-11-intermediate.h
EXTRA_OMP_CFLAGS += -DINTEL_OPENMP
endif # WANT_OPENMP_INTEL

if WANT_OPENMP_GNU
if WANT_OPENMP_GNU_4_2
WRAPPERS_OMP += wrappers/OMP/gnu-libgomp-4.2.c wrappers/OMP/gnu-libgomp-4.2.h
EXTRA_OMP_CFLAGS += -DGNU_OPENMP -DGNU_OPENMP_4_2
endif # WANT_OPENMP_GNU_4_2
if WANT_OPENMP_GNU_4_9
WRAPPERS_OMP += wrappers/OMP/gnu-libgomp-4.9.c wrappers/OMP/gnu-libgomp-4.9.h
EXTRA_OMP_CFLAGS += -DGNU_OPENMP -DGNU_OPENMP_4_9
endif # WANT_OPENMP_GNU_4_9
endif # WANT_OPENMP_GNU

if ARCH_POWERPC
if OS_LINUX
if WANT_OPENMP_IBM
 WRAPPERS_OMP += wrappers/OMP/ibm-xlsmp-1.6.c wrappers/OMP/ibm-xlsmp-1.6.h
 EXTRA_OMP_CFLAGS += -I/usr/src/linux
 EXTRA_OMP_CFLAGS += -DIBM_OPENMP
endif # ARCH_POWERPC
endif # OS_LINUX
endif # WANT_OPENMP_IBM

endif # WANT_OPENMP_OMPT

# Wrappers for TRT instrumentation (experimental)
WRAPPERS_TRT = wrappers/TRT/trt_wrapper.c wrappers/TRT/trt_wrapper.h \
 probes/TRT/trt_probe.c probes/TRT/trt_probe.h

# Wrappers for pthreads instrumentation
WRAPPERS_PTHREAD = wrappers/pthread/pthread_wrapper.c wrappers/pthread/pthread_wrapper.h \
 probes/pthread/pthread_probe.c probes/pthread/pthread_probe.h

if WANT_OPENSHMEM
WRAPPERS_OPENSHMEM = \
  wrappers/openshmem/openshmem_wrappers.c \
  wrappers/openshmem/openshmem_wrappers.h \
  probes/openshmem/openshmem_probes.c     \
  probes/openshmem/openshmem_probes.h
endif

if IS_CELL_MACHINE
WRAPPERS_CORE += wrappers/CELL/cell_wrapper.c wrappers/CELL/cell_wrapper.h
endif

core_MPI_SRCS = \
 hash_table.c hash_table.h                   \
 persistent_requests.c persistent_requests.h

core_PACX_SRCS = $(core_MPI_SRCS)

core_SRCS = \
 calltrace.c calltrace.h                     \
 signals.c signals.h                         \
 sampling.c sampling.h                       \
 xml-parse.c xml-parse.h                     \
 UF_gcc_instrument.c UF_gcc_instrument.h     \
 UF_xl_instrument.c UF_xl_instrument.h       \
 mode.c mode.h                               \
 threadid.h threadid.c                       \
 taskid.h taskid.c                           \
 defines.h                                   \
 $(COMMON_INC)/events.h                      \
 $(COMMON_INC)/record.h                      \
 $(COMMON_INC)/queue.h                       \
 $(COMMON_INC)/trace_mode.h                  \
 $(WRAPPERS_CORE)

# Auto instrumentation through constructor and destructors
auto_init_instr_SRCS = auto_init.c
auto_fini_instr_SRCS = auto_fini.c auto_fini.h

# CUDA trace
libcudatrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_CUDA) $(auto_init_instr_SRCS) $(auto_fini_instr_SRCS)
libcudampitrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_CUDA) $(WRAPPERS_MPI) $(core_MPI_SRCS) $(auto_fini_instr_SRCS)
libcudampitracef_la_SOURCES = $(libcudampitrace_la_SOURCES) mpif.h

# OpenCL trace
libocltrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_OCL) $(auto_init_instr_SRCS) $(auto_fini_instr_SRCS)
liboclmpitrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_OCL) $(WRAPPERS_MPI) $(core_MPI_SRCS) $(auto_fini_instr_SRCS)
liboclmpitracef_la_SOURCES = $(liboclmpitrace_la_SOURCES) mpif.h

# CUDA trace & MPItrace & OpenMP
libcudaompitrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_CUDA) $(WRAPPERS_MPI) $(WRAPPERS_OMP) $(core_MPI_SRCS)
libcudaompitracef_la_SOURCES = $(libcudaompitrace_la_SOURCES) mpif.h

# MPItrace & Sequential trace
libmpitrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_MPI) $(core_MPI_SRCS) $(auto_fini_instr_SRCS)
libmpitracef_la_SOURCES = $(libmpitrace_la_SOURCES) mpif.h
libseqtrace_la_SOURCES = $(core_SRCS) $(auto_init_instr_SRCS) $(auto_fini_instr_SRCS)

libptmpitrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_MPI) $(core_MPI_SRCS) $(WRAPPERS_PTHREAD) $(auto_fini_instr_SRCS)
libptmpitracef_la_SOURCES = $(libptmpitrace_la_SOURCES) mpif.h

# UPC (only C?)
libupctrace_la_SOURCES = $(core_SRCS) $(auto_init_instr_SRCS) $(auto_fini_instr_SRCS)

# PACXtrace 
libpacxtrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_PACX) $(core_PACX_SRCS)
libpacxtracef_la_SOURCES = $(libpacxtrace_la_SOURCES)

# OpenMP & OpenMP + MPI
libompitrace_la_SOURCES = $(libmpitrace_la_SOURCES) $(WRAPPERS_OMP)
libompitracef_la_SOURCES = $(libmpitracef_la_SOURCES) $(WRAPPERS_OMP) mpif.h
libomptrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_OMP) $(auto_init_instr_SRCS) $(auto_fini_instr_SRCS)

# SMPSS & SMPSS + MPI
libsmpsstrace_la_SOURCES = $(core_SRCS) $(auto_fini_instr_SRCS)
libsmpssmpitrace_la_SOURCES = $(libmpitrace_la_SOURCES)
libsmpssmpitracef_la_SOURCES = $(libmpitrace_la_SOURCES) mpif.h

# Nanos
libnanostrace_la_SOURCES = $(core_SRCS)
libnanosmpitrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_MPI) $(core_MPI_SRCS)
libnanosmpitracef_la_SOURCES = $(libnanosmpitrace_la_SOURCES) mpif.h

# pthreads & TRT
libpttrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_PTHREAD) $(auto_init_instr_SRCS) $(auto_fini_instr_SRCS)
libtrttrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_TRT) $(auto_init_instr_SRCS) $(auto_fini_instr_SRCS)

# OpenSHMEM
libopenshmemtrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_OPENSHMEM) $(auto_fini_instr_SRCS)

# LoadBalancing
libmpitrace_lb_la_SOURCES = $(libmpitrace_la_SOURCES)
libmpitracef_lb_la_SOURCES = $(libmpitrace_la_SOURCES) mpif.h
libompitrace_lb_la_SOURCES = $(libompitrace_la_SOURCES)
libompitracef_lb_la_SOURCES = $(libompitrace_la_SOURCES) mpif.h
libsmpssmpitrace_lb_la_SOURCES = $(libmpitrace_la_SOURCES)
libsmpssmpitracef_lb_la_SOURCES = $(libmpitrace_la_SOURCES) mpif.h
libnanosmpitrace_lb_la_SOURCES = $(libnanosmpitrace_la_SOURCES)
libnanosmpitracef_lb_la_SOURCES = $(libnanosmpitrace_la_SOURCES) mpif.h

# For DynInst instrumentation
lib_dyn_mpitracec_la_SOURCES = $(libmpitrace_la_SOURCES)
lib_dyn_mpitracef_la_SOURCES = $(libmpitracef_la_SOURCES)
lib_dyn_ompitracec_la_SOURCES = $(libompitrace_la_SOURCES)
lib_dyn_ompitracef_la_SOURCES = $(libompitracef_la_SOURCES)

lib_dyn_cudampitracec_la_SOURCES = $(libcudampitrace_la_SOURCES)
lib_dyn_cudampitracef_la_SOURCES = $(libcudampitracef_la_SOURCES)

lib_dyn_oclmpitracec_la_SOURCES = $(liboclmpitrace_la_SOURCES)
lib_dyn_oclmpitracef_la_SOURCES = $(liboclmpitracef_la_SOURCES)

core_MODULES = \
 $(COMMON_LIB)/libcommon.la \
 $(CLOCKS_LIB)/libclock.la \
 $(HWC_LIB)/libhwc.la

INTERFACE_API_PTHREAD = $(INTERFACES_LIB)/API/libiface_api_pthread.la
if PTHREAD_SUPPORT_IN_ALL_LIBS
 INTERFACE_API = $(INTERFACE_API_PTHREAD)
else
 INTERFACE_API = $(INTERFACES_LIB)/API/libiface_api.la
endif

# Apply these to BGQ and other non-BG machines

if IS_BG_MACHINE
if IS_BGQ_MACHINE
if WANT_MERGE_IN_TRACE
SEQUENTIAL_MERGE_LIB = $(MERGER_LIB)/libmpi2prv.la $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS)
SEQUENTIAL_CFLAGS_MERGE_LIB = -DEMBED_MERGE_IN_TRACE -I$(MERGER_INC)/common
if HAVE_MPI
PARALLEL_MERGE_LIB = $(MERGER_LIB)/parallel/libmpimpi2prv.la $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS)
PARALLEL_MERGE_LIB_AIX = $(MERGER_INC)/parallel/.libs/libmpimpi2prv.a $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS)
PARALLEL_CFLAGS_MERGE_LIB = -DEMBED_MERGE_IN_TRACE -I$(MERGER_INC)/common
endif
endif
endif
else
if WANT_MERGE_IN_TRACE
SEQUENTIAL_MERGE_LIB = $(MERGER_LIB)/libmpi2prv.la $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS)
SEQUENTIAL_CFLAGS_MERGE_LIB = -DEMBED_MERGE_IN_TRACE -I$(MERGER_INC)/common
if HAVE_MPI
PARALLEL_MERGE_LIB = $(MERGER_LIB)/parallel/libmpimpi2prv.la $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS)
PARALLEL_MERGE_LIB_AIX = $(MERGER_INC)/parallel/.libs/libmpimpi2prv.a $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS)
PARALLEL_CFLAGS_MERGE_LIB = -DEMBED_MERGE_IN_TRACE -I$(MERGER_INC)/common
endif
endif
endif

if HAVE_MPI
MPI_STATS_LIB = $(STATS_LIB)/libmpistats.la
UTILS_MPI_LIB = $(COMMON_MPI_DIR)/libcommon_mpi.la
ADDITIONAL_MPI_LIBS = $(MPI_STATS_LIB) $(UTILS_MPI_LIB)
endif

libcudatrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)
libcudampitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpi.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
libcudampitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpif.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)

libocltrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)
liboclmpitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpi.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
liboclmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpif.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)

libcudaompitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpi.la  $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
libcudaompitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpif.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)

libseqtrace_la_LIBADD  = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)
libmpitrace_la_LIBADD  = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpi.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
libmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpif.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
libptmpitrace_la_LIBADD  = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(INTERFACES_LIB)/MPI/libiface_mpi.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
libptmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(INTERFACES_LIB)/MPI/libiface_mpif.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)

libupctrace_la_LIBADD  = $(core_MODULES) $(INTERFACE_API)

libpacxtrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/PACX/libiface_pacx.la
libpacxtracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/PACX/libiface_pacxf.la

libompitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpi.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
libompitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpif.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
libomptrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)

libsmpssmpitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpi.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
libsmpssmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpif.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
libsmpsstrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)

libnanostrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)
libnanosmpitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpi.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
libnanosmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpif.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)

libpttrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(SEQUENTIAL_MERGE_LIB)
libtrttrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)

libopenshmemtrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)

libmpitrace_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpi_lb.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
libmpitracef_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpif_lb.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
libompitrace_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpi_lb.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
libompitracef_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpif_lb.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)

libsmpssmpitrace_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpi_lb.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
libsmpssmpitracef_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpif_lb.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)

libnanosmpitrace_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpi_lb.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
libnanosmpitracef_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/libiface_mpif_lb.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)

lib_dyn_mpitracec_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/lib_dyn_iface_mpic.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
lib_dyn_mpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/lib_dyn_iface_mpif.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
lib_dyn_ompitracec_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/lib_dyn_iface_mpic.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
lib_dyn_ompitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/lib_dyn_iface_mpif.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
lib_dyn_cudampitracec_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/lib_dyn_iface_mpic.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)
lib_dyn_cudampitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_LIB)/MPI/lib_dyn_iface_mpif.la $(PARALLEL_MERGE_LIB) $(ADDITIONAL_MPI_LIBS)

#
# Compilation flags
#
core_INCLUDES = \
 -I$(TRACER_INC) -I$(HWC_INC) -I$(CLOCKS_INC) -I$(INTERFACES_INC)/API -I$(INTERFACES_INC)/MPI -I$(INTERFACES_INC)/PACX \
 -I$(STATS_INC)/MPI \
 -I$(WRAPPERS_INC)/API -I$(WRAPPERS_INC)/MPI -I$(WRAPPERS_INC)/PACX \
 -I$(WRAPPERS_INC)/OMP -I$(WRAPPERS_INC)/pthread \
 -I$(WRAPPERS_INC)/CUDA -I$(WRAPPERS_INC)/OPENCL \
 -I$(WRAPPERS_INC)/fork -I$(PROBES_INC)/fork \
 -I$(WRAPPERS_INC)/MALLOC -I$(PROBES_INC)/MALLOC \
 -I$(WRAPPERS_INC)/IO -I$(PROBES_INC)/IO \
 -I$(WRAPPERS_INC)/openshmem \
 -I$(top_srcdir) -I$(top_srcdir)/include -I$(COMMON_INC) $(UNWIND_CFLAGS)

if IS_CELL_MACHINE
  core_INCLUDES += -I$(WRAPPERS_INC)/CELL -I$(TRACER_INC)/spu
endif
if HAVE_PAPI
if !HAVE_PAPI_EMBEDDED
  core_INCLUDES += @PAPI_CFLAGS@
endif
endif
if HAVE_XML2
  core_INCLUDES += @XML2_CFLAGS@
endif
if WANT_OPENMP
  core_INCLUDES += -I$(PROBES_INC)/OMP
endif
if WANT_PTHREAD
  core_INCLUDES += -I$(PROBES_INC)/pthread
endif
if WANT_TRT
  core_INCLUDES += -I$(PROBES_INC)/TRT
endif
if WANT_CUDAorCUPTI
  core_INCLUDES += -I$(PROBES_INC)/CUDA
endif
if WANT_OPENCL
  core_INCLUDES += -I$(PROBES_INC)/OPENCL
endif
if WANT_CUPTI
  core_INCLUDES += -I$(CUPTI_DIR)/include
endif
if WANT_OPENSHMEM
  core_INCLUDES += -I$(PROBES_INC)/openshmem
endif

if HAVE_PAPI
if !HAVE_PAPI_EMBEDDED
  PAPI_LINKER_FLAGS = -L@PAPI_SHAREDLIBSDIR@ -R @PAPI_SHAREDLIBSDIR@ @PAPI_LIBS@
endif
endif
if HAVE_PMAPI
  PMAPI_LINKER_FLAGS = -lpmapi
endif
HWC_LINKER_FLAGS = $(PAPI_LINKER_FLAGS) $(PMAPI_LINKER_FLAGS)

if HAVE_XML2
  XML2_LINKER_FLAGS = -L@XML2_SHAREDLIBSDIR@ -R @XML2_SHAREDLIBSDIR@ @XML2_LIBS@
endif
if HAVE_LIBDL
  LDL = -ldl
endif
if HAVE_UNWIND
  UNWIND_LINKER_FLAGS = -L@UNWIND_LIBSDIR@ -R @UNWIND_LIBSDIR@ @UNWIND_LIBS@
endif

# POSIX clock may be available by default. Avoid adding these if unneeded.
if USE_POSIX_CLOCK
if NEED_POSIX_CLOCK 
  POSIXCLOCK_LINKER_FLAGS = -L@RT_LIBSDIR@ -R @RT_LIBSDIR@ @RT_LIBS@
endif
endif

if HAVE_ZLIB
  LIBZ_LINKER_FLAGS = -L@LIBZ_SHAREDLIBSDIR@ -R @LIBZ_SHAREDLIBSDIR@ @LIBZ_LIBS@
endif

core_CFLAGS =
if PTHREAD_SUPPORT_IN_ALL_LIBS
core_CFLAGS += -DPTHREAD_SUPPORT
endif
if OS_ANDROID
core_CFLAGS += -DHOST_NAME_MAX=255
endif

if HAVE_MPI
EXTRA_MPI_CFLAGS= -I$(COMMON_MPI_DIR) @MPI_CFLAGS@ $(PARALLEL_CFLAGS_MERGE_LIB)
if SINGLE_MPI_LIBRARY
MPITRACE_SYMBOLS= -DC_SYMBOLS -DFORTRAN_SYMBOLS
else
MPITRACE_SYMBOLS= -DC_SYMBOLS
endif
endif

COMMON_LINKER_FLAGS = -release $(PACKAGE_VERSION) $(HWC_LINKER_FLAGS) $(UNWIND_LINKER_FLAGS) $(POSIXCLOCK_LINKER_FLAGS) $(MRNET_LINKER_FLAGS) $(XML2_LINKER_FLAGS) $(LDL) $(LIBZ_LINKER_FLAGS)
#COMMON_LINKER_FLAGS = -avoid-version $(HWC_LINKER_FLAGS) $(UNWIND_LINKER_FLAGS) $(POSIXCLOCK_LINKER_FLAGS) $(MRNET_LINKER_FLAGS) $(XML2_LINKER_FLAGS) $(LDL) $(LIBZ_LINKER_FLAGS)

libcudatrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
libcudampitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(MPITRACE_SYMBOLS) -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
libcudampitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DFORTRAN_SYMBOLS -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)

libocltrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS)
liboclmpitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(MPITRACE_SYMBOLS) -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS) -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS)
liboclmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DFORTRAN_SYMBOLS -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS) -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS)

libcudaompitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) $(MPITRACE_SYMBOLS) -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS) $(EXTRA_OMP_CFLAGS) -DOMP_SUPPORT
libcudaompitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) -DFORTRAN_SYMBOLS -DMPI_SUPPORT -DOMP_SUPPORT $(EXTRA_MPI_CFLAGS) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)

libseqtrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB)

libupctrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) -DUPC_SUPPORT

libmpitrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) $(MPITRACE_SYMBOLS) -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS)
libmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DFORTRAN_SYMBOLS -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS)

libptmpitrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) $(MPITRACE_SYMBOLS) -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS) -DPTHREAD_SUPPORT
libptmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DFORTRAN_SYMBOLS -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS) -DPTHREAD_SUPPORT

libpacxtrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) -I$(COMMON_MPI_DIR) @PACX_CFLAGS@ $(MPITRACE_SYMBOLS) -DPACX_SUPPORT
libpacxtracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -I$(COMMON_MPI_DIR) @PACX_CFLAGS@ -DFORTRAN_SYMBOLS -DPACX_SUPPORT

libompitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) $(MPITRACE_SYMBOLS) -DOMP_SUPPORT -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS)
libompitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) -DFORTRAN_SYMBOLS -DOMP_SUPPORT -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS)
libomptrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) -DOMP_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)

libsmpssmpitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(MPITRACE_SYMBOLS) -DSMPSS_SUPPORT -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS)
libsmpssmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DFORTRAN_SYMBOLS -DSMPSS_SUPPORT -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS)
libsmpsstrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DSMPSS_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)

libnanostrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DNANOS_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)
libnanosmpitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(MPITRACE_SYMBOLS) -DNANOS_SUPPORT -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS)
libnanosmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DNANOS_SUPPORT -DFORTRAN_SYMBOLS -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS)

libpttrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DPTHREAD_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)
libtrttrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DTRT_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)

libopenshmemtrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @OPENSHMEM_CFLAGS@ -DOPENSHMEM_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)

libmpitrace_lb_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) $(MPITRACE_SYMBOLS) -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS)
libmpitracef_lb_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) -DFORTRAN_SYMBOLS -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS)
libompitrace_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) $(MPITRACE_SYMBOLS) -DOMP_SUPPORT -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS)
libompitracef_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) -DFORTRAN_SYMBOLS -DOMP_SUPPORT -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS)

libsmpssmpitrace_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(MPITRACE_SYMBOLS) -DSMPSS_SUPPORT -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS)
libsmpssmpitracef_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DFORTRAN_SYMBOLS -DSMPSS_SUPPORT -DMPI_SUPPORT $(EXTRA_MPI_CFLAGS)

libnanosmpitrace_lb_la_CFLAGS = $(libnanosmpitrace_la_CFLAGS)
libnanosmpitracef_lb_la_CFLAGS = $(libnanosmpitracef_la_CFLAGS)

lib_dyn_mpitracec_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(MPITRACE_SYMBOLS) -DMPI_SUPPORT -DDYNINST_MODULE $(EXTRA_MPI_CFLAGS)
lib_dyn_mpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(MPITRACE_SYMBOLS) -DFORTRAN_SYMBOLS -DMPI_SUPPORT -DDYNINST_MODULE $(EXTRA_MPI_CFLAGS)
lib_dyn_ompitracec_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) $(MPITRACE_SYMBOLS) -DMPI_SUPPORT -DDYNINST_MODULE -DOMP_SUPPORT $(EXTRA_MPI_CFLAGS)
lib_dyn_ompitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) $(MPITRACE_SYMBOLS) -DFORTRAN_SYMBOLS -DMPI_SUPPORT -DDYNINST_MODULE -DOMP_SUPPORT $(EXTRA_MPI_CFLAGS)
lib_dyn_cudampitracec_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) $(MPITRACE_SYMBOLS) -DMPI_SUPPORT -DDYNINST_MODULE -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS) $(EXTRA_MPI_CFLAGS)
lib_dyn_cudampitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) $(MPITRACE_SYMBOLS) -DFORTRAN_SYMBOLS -DMPI_SUPPORT -DDYNINST_MODULE -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS) $(EXTRA_MPI_CFLAGS)

lib_dyn_oclmpitracec_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) $(MPITRACE_SYMBOLS) -DMPI_SUPPORT -DDYNINST_MODULE -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS) $(EXTRA_MPI_CFLAGS)
lib_dyn_oclmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) $(MPITRACE_SYMBOLS) -DFORTRAN_SYMBOLS -DMPI_SUPPORT -DDYNINST_MODULE -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS) $(EXTRA_MPI_CFLAGS)

#
# Online support is only activated for the MPI libraries
# 
if HAVE_ONLINE
libmpitrace_la_LIBADD       += $(ONLINE_LIB)/libonline.la
libmpitracef_la_LIBADD      += $(ONLINE_LIB)/libonline.la
libompitrace_la_LIBADD      += $(ONLINE_LIB)/libonline.la
libompitracef_la_LIBADD     += $(ONLINE_LIB)/libonline.la
lib_dyn_mpitracec_la_LIBADD += $(ONLINE_LIB)/libonline.la
lib_dyn_mpitracef_la_LIBADD += $(ONLINE_LIB)/libonline.la

libmpitrace_la_CFLAGS       += -DHAVE_ONLINE -I$(ONLINE_INC)
libmpitracef_la_CFLAGS      += -DHAVE_ONLINE -I$(ONLINE_INC)
libompitrace_la_CFLAGS      += -DHAVE_ONLINE -I$(ONLINE_INC)
libompitracef_la_CFLAGS     += -DHAVE_ONLINE -I$(ONLINE_INC)
lib_dyn_mpitracec_la_CFLAGS += -DHAVE_ONLINE -I$(ONLINE_INC)
lib_dyn_mpitracef_la_CFLAGS += -DHAVE_ONLINE -I$(ONLINE_INC)
endif

#
# Link flags
#
if HAVE_MPI_WITH_SHARED_LIBS
NEW_MPI_LIBS = -L@MPI_SHAREDLIBSDIR@ -R @MPI_SHAREDLIBSDIR@ @MPI_LIBS@
else
NEW_MPI_LIBS = -L@MPI_LIBSDIR@ @MPI_LIBS@ -static
endif

libcudaompitrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libcudaompitracef_la_LDFLAGS = $(COMMON_LINKER_FLAGS)

libseqtrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)

libupctrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)

libmpitrace_la_LDFLAGS = $(NEW_MPI_LIBS) $(COMMON_LINKER_FLAGS)
libmpitracef_la_LDFLAGS = $(libmpitrace_la_LDFLAGS) # -static 

libptmpitrace_la_LDFLAGS = $(NEW_MPI_LIBS) $(COMMON_LINKER_FLAGS) $(LIBPTHREAD)
libptmpitracef_la_LDFLAGS = $(libptmpitrace_la_LDFLAGS) # -static 

libpacxtrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libpacxtracef_la_LDFLAGS = $(libpacxtrace_la_LDFLAGS) # -static

libcudatrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libcudampitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libcudampitracef_la_LDFLAGS = $(libmpitracef_la_LDFLAGS)

libocltrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
liboclmpitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
liboclmpitracef_la_LDFLAGS = $(libmpitracef_la_LDFLAGS)

libomptrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libompitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libompitracef_la_LDFLAGS = $(libmpitracef_la_LDFLAGS)

libsmpsstrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libsmpssmpitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libsmpssmpitracef_la_LDFLAGS = $(libmpitracef_la_LDFLAGS)

libnanostrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libnanosmpitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libnanosmpitracef_la_LDFLAGS = $(libmpitracef_la_LDFLAGS)

libpttrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS) $(LIBPTHREAD)
libtrttrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)

libopenshmemtrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS) $(OPENSHMEM_LDFLAGS)

libmpitrace_lb_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libmpitracef_lb_la_LDFLAGS = $(libmpitracef_la_LDFLAGS) # -static
libompitrace_lb_la_LDFLAGS = $(libompitrace_la_LDFLAGS)
libompitracef_lb_la_LDFLAGS = $(libompitracef_la_LDFLAGS)

libsmpssmpitrace_lb_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libsmpssmpitracef_lb_la_LDFLAGS = $(libmpitrace_lb_la_LDFLAGS) # -static

libnanosmpitrace_lb_la_LDFLAGS = $(libnanosmpitrace_la_LDFLAGS)
libnanosmpitracef_lb_la_LDFLAGS = $(libnanosmpitracef_la_LDFLAGS)

lib_dyn_mpitracec_la_LDFLAGS = $(libmpitrace_la_LDFLAGS) -shared
lib_dyn_mpitracef_la_LDFLAGS = $(libmpitrace_la_LDFLAGS) -shared
lib_dyn_ompitracec_la_LDFLAGS = $(lib_dyn_mpitracec_la_LDFLAGS) # inherits -shared
lib_dyn_ompitracef_la_LDFLAGS = $(lib_dyn_mpitracef_la_LDFLAGS) # inherits -shared
lib_dyn_cudampitracef_la_LDFLAGS = $(lib_dyn_mpitracef_la_LDFLAGS) # inherits -shared
lib_dyn_cudampitracec_la_LDFLAGS = $(lib_dyn_mpitracec_la_LDFLAGS) # inherits -shared

## -static tells libtool to generate ONLY the static library (we don't want
##   the libmpitracef.so because it don't preloads fine with programs)
## -shared tells libtool to generate ONLY shared libraries (right now,
##   only for dyninst modules)
## -R flag works like -Wl,-rpath
## -no-undefined could be interesting for AIX?
##   check http://www.andamooka.org/reader.pl?pgid=autobookautobook_89

WRAPPERS_AIX_CORE = \
 wrappers/API/buffers.c \
 wrappers/API/wrapper.c \
 wrappers/API/misc_wrapper.c

core_AIX_SRCS = \
 calltrace.c \
 signals.c \
 xml-parse.c \
 UF_gcc_instrument.c \
 UF_xl_instrument.c \
 mode.c \
 threadid.c \
 taskid.c \
 wrappers/MPI/mpi_wrapper.c \
 hash_table.c  \
 persistent_requests.c \
 $(WRAPPERS_AIX_CORE)

if OS_AIX
if HAVE_MPI
if HAVE_PAPI
libmpitrace.so: $(core_AIX_SRCS)
	xlc -G -I ../.. $(core_INCLUDES) $(core_CFLAGS) -I/usr/lpp/ppe.poe/include $(MPITRACE_SYMBOLS) -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB) \
	$(core_AIX_SRCS) \
	$(PARALLEL_MERGE_LIB_AIX) \
	../common/.libs/libcommon.a \
	../common/MPI/.libs/libcommon_mpi.a \
	.libs/libmpistats.a \
	clocks/.libs/libclock.a \
	hwc/.libs/libhwc.a \
	interfaces/API/.libs/libiface_api.a \
	interfaces/MPI/.libs/libiface_mpi.a \
	$(MPI_LDFLAGS) $(MPI_LIBS) \
	$(PAPI_LDFLAGS) $(PAPI_LIBS) \
	$(XML_LIBS) -lz \
	-o libmpitrace.so
else
libmpitrace.so: $(core_AIX_SRCS)
	xlc -G -I ../.. $(core_INCLUDES) $(core_CFLAGS) -I/usr/lpp/ppe.poe/include $(MPITRACE_SYMBOLS) -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB) \
	$(core_AIX_SRCS) \
	$(PARALLEL_MERGE_LIB_AIX) \
	../common/.libs/libcommon.a \
	../common/MPI/.libs/libcommon_mpi.a \
	.libs/libmpistats.a \
	clocks/.libs/libclock.a \
	hwc/.libs/libhwc.a \
	interfaces/API/.libs/libiface_api.a \
	interfaces/MPI/.libs/libiface_mpi.a \
	$(MPI_LDFLAGS) $(MPI_LIBS) \
	$(XML_LIBS) -lz \
	-o libmpitrace.so
endif
endif
endif

install-exec-hook:
if IS_CELL_MACHINE
	cd $(libdir); $(LN_S) -f libseqtrace.a libpputrace.a
endif
if OS_AIX
if HAVE_MPI
	gmake libmpitrace.so; cp libmpitrace.so $(libdir)
endif
endif

## These are the options of the previous Makefile (from Imakefile) which are
## not still ported to the configure system!

#### -DDEBUG_COMMUNICATORS mostra informacio sobre els comunicadors.
####
#### -DDEBUG_MPITRACE Prints debug information of the "internals"

######################################################################################
###                               SEPARATE MODULES                                 ###
######################################################################################

#libextrae_core_la_SOURCES = $(core_SRCS) $(auto_init_instr_SRCS) $(auto_fini_instr_SRCS)
#libextrae_core_la_LIBADD  = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)
#libextrae_core_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DSTANDALONE
#libextrae_core_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
#
#libextrae_mpi_la_SOURCES = $(WRAPPERS_MPI) $(core_MPI_SRCS)
#libextrae_mpi_la_LIBADD  = $(INTERFACES_LIB)/MPI/libiface_mpi.la $(PARALLEL_MERGE_LIB) $(MPI_STATS_LIB)
#libextrae_mpi_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ $(MPITRACE_SYMBOLS) -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB) -DSTANDALONE
#if HAVE_ONLINE
#libextrae_mpi_la_LIBADD += $(ONLINE_LIm)/libonline.la
#libextrae_mpi_la_CFLAGS += -DHAVE_ONLINE -I$(ONLINE_INC)
#endif
#libextrae_mpi_la_LDFLAGS = $(NEW_MPI_LIBS) $(COMMON_LINKER_FLAGS)
#
#libextrae_openmp_la_SOURCES = $(WRAPPERS_OMP) 
#libextrae_openmp_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) -DOMP_SUPPORT -DSTANDALONE
#libextrae_openmp_la_LDFLAGS = $(COMMON_LINKER_FLAGS)

