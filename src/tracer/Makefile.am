include $(top_srcdir)/PATHS

SUBDIRS = clocks hwc interfaces wrappers sampling

if HAVE_ONLINE
  SUBDIRS += online
endif

if HAVE_CUDA
WRAPPERS_CUDA_CFLAGS = @CUDA_CFLAGS@
endif

if HAVE_CUPTI
WRAPPERS_CUDA_CFLAGS = @CUDA_CFLAGS@ @CUPTI_CFLAGS@
endif

if WANT_OPENACC
WRAPPERS_OPENACC_CFLAGS = @OPENACC_CFLAGS@
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

if HAVE_CUDA
  lib_LTLIBRARIES += libcudatrace.la
if WANT_PTHREAD
  lib_LTLIBRARIES += libptcudatrace.la
endif
if WANT_OPENMP
  lib_LTLIBRARIES += libcudaomptrace.la libcudanewomptrace.la
endif
endif

if WANT_OPENCL
  lib_LTLIBRARIES += libocltrace.la
endif

if WANT_OPENACC
  lib_LTLIBRARIES += liboacctrace.la
if HAVE_CUDA
    lib_LTLIBRARIES += liboacccudatrace.la
endif
endif

if HAVE_MPI
  lib_LTLIBRARIES += libmpitrace.la libmpitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += libmpitracecf.la
endif
if WANT_OPENMP
  lib_LTLIBRARIES += libompitrace.la libompitracef.la libnewompitrace.la libnewompitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += libompitracecf.la libnewompitracecf.la
endif
endif
if HAVE_CUDA
  lib_LTLIBRARIES += libcudampitrace.la libcudampitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += libcudampitracecf.la
endif
if WANT_PTHREAD
  lib_LTLIBRARIES += libptcudampitrace.la
  lib_LTLIBRARIES += libptcudampitracef.la
endif

endif
if WANT_SMPSS
  lib_LTLIBRARIES += libsmpssmpitrace.la libsmpssmpitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += libsmpssmpitracecf.la
endif
endif
if WANT_OPENCL
  lib_LTLIBRARIES += liboclmpitrace.la liboclmpitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += liboclmpitracecf.la
endif
endif
if WANT_NANOS
endif
endif

if HAVE_GASPI
  lib_LTLIBRARIES += libgaspitrace.la
if HAVE_MPI
  lib_LTLIBRARIES += libmpigaspitrace.la
endif
if WANT_OPENMP
  lib_LTLIBRARIES += libompgaspitrace.la
if HAVE_MPI
  lib_LTLIBRARIES += libompigaspitrace.la
endif
endif
if WANT_PTHREAD
  lib_LTLIBRARIES += libptgaspitrace.la
if HAVE_MPI
  lib_LTLIBRARIES += libptmpigaspitrace.la
endif
endif
if HAVE_CUDA
  lib_LTLIBRARIES += libcudagaspitrace.la
if HAVE_MPI
  lib_LTLIBRARIES += libcudampigaspitrace.la
endif
endif
if WANT_OPENCL
  lib_LTLIBRARIES += liboclgaspitrace.la
if HAVE_MPI
  lib_LTLIBRARIES += liboclmpigaspitrace.la
endif
endif
if WANT_NANOS
  lib_LTLIBRARIES += libnanosgaspitrace.la
if HAVE_MPI
  lib_LTLIBRARIES += libnanosmpigaspitrace.la
endif
endif
endif

if HAVE_MPI
if WANT_OPENMP
if HAVE_CUDA
  lib_LTLIBRARIES += libcudaompitrace.la libcudaompitracef.la libcudanewompitrace.la libcudanewompitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += libcudaompitracecf.la libcudanewompitracecf.la
endif
endif
endif
endif

if HAVE_MPI
if WANT_OPENACC
  lib_LTLIBRARIES += liboaccmpitrace.la liboaccmpitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += liboaccmpitracecf.la
endif
if HAVE_CUDA
  lib_LTLIBRARIES += liboacccudampitrace.la liboacccudampitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += liboacccudampitracecf.la
endif
endif
endif
endif

if WANT_OPENMP
  lib_LTLIBRARIES += libomptrace.la libnewomptrace.la
endif
if WANT_SMPSS
  lib_LTLIBRARIES += libsmpsstrace.la
endif
if WANT_NANOS
  lib_LTLIBRARIES += libnanostrace.la
if HAVE_MPI
    lib_LTLIBRARIES += libnanosmpitrace.la libnanosmpitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += libnanosmpitracecf.la
endif
endif
endif
if WANT_PTHREAD
  lib_LTLIBRARIES += libpttrace.la
if HAVE_MPI
  lib_LTLIBRARIES += libptmpitrace.la libptmpitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += libptmpitracecf.la
endif
endif
endif

if WANT_OPENSHMEM
  lib_LTLIBRARIES += libopenshmemtrace.la
endif

if HAVE_DYNINST
if HAVE_MPI
  lib_LTLIBRARIES += lib_dyn_mpitracec.la lib_dyn_mpitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += lib_dyn_mpitracecf.la
endif
if WANT_OPENMP
  lib_LTLIBRARIES += lib_dyn_ompitracec.la lib_dyn_ompitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += lib_dyn_ompitracecf.la
endif
endif
if HAVE_CUDA
  lib_LTLIBRARIES += lib_dyn_cudampitracec.la lib_dyn_cudampitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += lib_dyn_cudampitracecf.la
endif
endif
if WANT_OPENCL
  lib_LTLIBRARIES += lib_dyn_oclmpitracec.la lib_dyn_oclmpitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += lib_dyn_oclmpitracecf.la
endif
endif
endif
endif

# Auto instrumentation through constructor and destructors
auto_init_instr_SRCS = auto_init.c
auto_fini_instr_SRCS = auto_fini.c auto_fini.h

WRAPPERS_CORE = \
 wrappers/API/buffers.c wrappers/API/buffers.h \
 wrappers/API/wrapper.c wrappers/API/wrapper.h \
 wrappers/API/misc_wrapper.c wrappers/API/misc_wrapper.h \
 wrappers/API/threadinfo.c wrappers/API/threadinfo.h \
 wrappers/API/trace_buffers.h \
 trace_macros.h trace_macros_mpi.h trace_macros_omp.h \
 trace_macros_gpu.h trace_hwc.h

core_SRCS = \
 calltrace.c calltrace.h                     \
 signals.c signals.h                         \
 xml-parse.c xml-parse.h                     \
 change_mode.c change_mode.h                 \
 UF_gcc_instrument.c UF_gcc_instrument.h     \
 UF_xl_instrument.c UF_xl_instrument.h       \
 threadid.h threadid.c                       \
 taskid.h taskid.c                           \
 xalloc.c                                    \
 defines.h                                   \
 $(COMMON_INC)/events.h                      \
 $(COMMON_INC)/record.h                      \
 $(COMMON_INC)/queue.h                       \
 $(COMMON_INC)/trace_mode.h                  \
 $(auto_init_instr_SRCS)                     \
 $(auto_fini_instr_SRCS)                     \
 $(WRAPPERS_CORE)

burst_mode_SRCS = $(STATS_DIR)/stats_types.h \
 $(STATS_DIR)/stats_module.c $(STATS_DIR)/stats_module.h \
 $(BURST_MODE_DIR)/burst_mode.c $(BURST_MODE_DIR)/burst_mode.h

mpi_stats_SRCS = \
 $(STATS_DIR)/MPI/mpi_stats.c $(STATS_DIR)/MPI/mpi_stats.h

omp_stats_SRCS = \
 $(STATS_DIR)/OMP/omp_utils.c $(STATS_DIR)/OMP/omp_utils.h \
 $(STATS_DIR)/OMP/omp_stats.c $(STATS_DIR)/OMP/omp_stats.h

# OPENACC trace
liboacctrace_la_SOURCES = $(core_SRCS)
liboacccudatrace_la_SOURCES = $(liboacctrace_la_SOURCES) $(WRAPPERS_CUDA)
liboaccmpitrace_la_SOURCES = $(liboacctrace_la_SOURCES) $(mpi_stats_SRCS)
liboaccmpitracef_la_SOURCES = $(liboaccmpitrace_la_SOURCES) extrae_mpif.h
liboacccudampitrace_la_SOURCES = $(liboacctrace_la_SOURCES) $(WRAPPERS_CUDA) $(mpi_stats_SRCS)
liboacccudampitracef_la_SOURCES = $(liboacccudampitrace_la_SOURCES) extrae_mpif.h
if SINGLE_MPI_LIBRARY
  liboaccmpitracecf_la_SOURCES = $(liboaccmpitracef_la_SOURCES)
  liboacccudampitracecf_la_SOURCES = $(liboacccudampitracef_la_SOURCES)
endif

# GASPI trace
libgaspitrace_la_SOURCES = $(core_SRCS)
libmpigaspitrace_la_SOURCES = $(libgaspitrace_la_SOURCES) $(mpi_stats_SRCS)
libompigaspitrace_la_SOURCES = $(libmpigaspitrace_la_SOURCES)
libompgaspitrace_la_SOURCES = $(libgaspitrace_la_SOURCES)
libptmpigaspitrace_la_SOURCES = $(libmpigaspitrace_la_SOURCES)
libptgaspitrace_la_SOURCES = $(libgaspitrace_la_SOURCES)
libcudampigaspitrace_la_SOURCES = $(libmpigaspitrace_la_SOURCES) $(WRAPPERS_CUDA)
libcudagaspitrace_la_SOURCES = $(libgaspitrace_la_SOURCES) $(WRAPPERS_CUDA)
liboclmpigaspitrace_la_SOURCES = $(libmpigaspitrace_la_SOURCES) $(WRAPPERS_OCL)
liboclgaspitrace_la_SOURCES = $(libgaspitrace_la_SOURCES) $(WRAPPERS_OCL)
libnanosmpigaspitrace_la_SOURCES = $(libmpigaspitrace_la_SOURCES)
libnanosgaspitrace_la_SOURCES = $(libgaspitrace_la_SOURCES)

# CUDA trace
libcudatrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_CUDA)
libcudampitrace_la_SOURCES = $(libcudatrace_la_SOURCES) $(mpi_stats_SRCS)
libcudampitracef_la_SOURCES = $(libcudampitrace_la_SOURCES) extrae_mpif.h
libcudaomptrace_la_SOURCES = $(libcudatrace_la_SOURCES)
libcudanewomptrace_la_SOURCES = $(libcudatrace_la_SOURCES) $(omp_stats_SRCS)
libptcudatrace_la_SOURCES = $(libcudatrace_la_SOURCES)
libptcudampitrace_la_SOURCES = $(libcudampitrace_la_SOURCES)
libptcudampitracef_la_SOURCES = $(libcudampitracef_la_SOURCES)

if SINGLE_MPI_LIBRARY
  libcudampitracecf_la_SOURCES = $(libcudampitracef_la_SOURCES)
endif

# OpenCL trace
libocltrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_OCL)
liboclmpitrace_la_SOURCES = $(libocltrace_la_SOURCES) $(mpi_stats_SRCS)
liboclmpitracef_la_SOURCES = $(liboclmpitrace_la_SOURCES) extrae_mpif.h

if SINGLE_MPI_LIBRARY
  liboclmpitracecf_la_SOURCES = $(liboclmpitracef_la_SOURCES)
endif

# CUDA trace & MPItrace & OpenMP
libcudaompitrace_la_SOURCES = $(libcudampitrace_la_SOURCES)
libcudaompitracef_la_SOURCES = $(libcudampitracef_la_SOURCES)
libcudanewompitrace_la_SOURCES = $(libcudatrace_la_SOURCES) $(mpi_stats_SRCS) $(omp_stats_SRCS)
libcudanewompitracef_la_SOURCES = $(libcudanewompitrace_la_SOURCES) extrae_mpif.h

if SINGLE_MPI_LIBRARY
  libcudaompitracecf_la_SOURCES = $(libcudaompitracef_la_SOURCES)
  libcudanewompitracecf_la_SOURCES = $(libcudanewompitracef_la_SOURCES)
endif

# MPItrace & Sequential trace
libmpitrace_la_SOURCES = $(core_SRCS) $(mpi_stats_SRCS) $(burst_mode_SRCS)
libmpitracef_la_SOURCES = $(libmpitrace_la_SOURCES) extrae_mpif.h
libseqtrace_la_SOURCES = $(core_SRCS)

if SINGLE_MPI_LIBRARY
  libmpitracecf_la_SOURCES = $(libmpitracef_la_SOURCES)
endif

libptmpitrace_la_SOURCES = $(core_SRCS) $(mpi_stats_SRCS)
libptmpitracef_la_SOURCES = $(libptmpitrace_la_SOURCES) extrae_mpif.h

if SINGLE_MPI_LIBRARY
  libptmpitracecf_la_SOURCES = $(libptmpitracef_la_SOURCES)
endif

# OpenMP & OpenMP + MPI
libompitrace_la_SOURCES = $(core_SRCS) $(mpi_stats_SRCS)
libompitracef_la_SOURCES = $(libompitrace_la_SOURCES) extrae_mpif.h
libomptrace_la_SOURCES = $(core_SRCS)
libnewomptrace_la_SOURCES = $(core_SRCS) $(omp_stats_SRCS) $(burst_mode_SRCS)
libnewompitrace_la_SOURCES = $(libnewomptrace_la_SOURCES) $(mpi_stats_SRCS)
libnewompitracef_la_SOURCES = $(libnewompitrace_la_SOURCES)  extrae_mpif.h

if SINGLE_MPI_LIBRARY
  libompitracecf_la_SOURCES = $(libompitracef_la_SOURCES)
  libnewompitracecf_la_SOURCES = $(libnewompitracef_la_SOURCES)
endif

# SMPSS & SMPSS + MPI
libsmpsstrace_la_SOURCES = $(core_SRCS)
libsmpssmpitrace_la_SOURCES = $(libsmpsstrace_la_SOURCES) $(mpi_stats_SRCS)
libsmpssmpitracef_la_SOURCES = $(libsmpssmpitrace_la_SOURCES) extrae_mpif.h

if SINGLE_MPI_LIBRARY
  libsmpssmpitracecf_la_SOURCES = $(libsmpssmpitracef_la_SOURCES)
endif

# Nanos
libnanostrace_la_SOURCES = $(core_SRCS)
libnanosmpitrace_la_SOURCES = $(libnanostrace_la_SOURCES) $(mpi_stats_SRCS)
libnanosmpitracef_la_SOURCES = $(libnanosmpitrace_la_SOURCES) extrae_mpif.h

if SINGLE_MPI_LIBRARY
  libnanosmpitracecf_la_SOURCES = $(libnanosmpitracef_la_SOURCES)
endif

# pthreads
libpttrace_la_SOURCES = $(core_SRCS)

# OpenSHMEM
libopenshmemtrace_la_SOURCES = $(core_SRCS)

# For DynInst instrumentation
lib_dyn_mpitracec_la_SOURCES = $(libmpitrace_la_SOURCES)
lib_dyn_mpitracef_la_SOURCES = $(libmpitracef_la_SOURCES)
lib_dyn_mpitracecf_la_SOURCES = $(libmpitracecf_la_SOURCES)

lib_dyn_ompitracec_la_SOURCES = $(libompitrace_la_SOURCES)
lib_dyn_ompitracef_la_SOURCES = $(libompitracef_la_SOURCES)
lib_dyn_ompitracecf_la_SOURCES = $(libompitracecf_la_SOURCES)

lib_dyn_cudampitracec_la_SOURCES = $(libcudampitrace_la_SOURCES)
lib_dyn_cudampitracef_la_SOURCES = $(libcudampitracef_la_SOURCES)
lib_dyn_cudampitracecf_la_SOURCES = $(libcudampitracecf_la_SOURCES)

lib_dyn_oclmpitracec_la_SOURCES = $(liboclmpitrace_la_SOURCES)
lib_dyn_oclmpitracef_la_SOURCES = $(liboclmpitracef_la_SOURCES)
lib_dyn_oclmpitracecf_la_SOURCES = $(liboclmpitracecf_la_SOURCES)

core_MODULES = \
 $(COMMON_LIB)/libcommon.la \
 $(CLOCKS_LIB)/libclock.la \
 $(HWC_LIB)/libhwc.la \
 $(WRAPPERS_LIB)/MALLOC/libwrap_malloc.la \
 $(WRAPPERS_LIB)/IO/libwrap_io.la \
 $(WRAPPERS_LIB)/SYSCALL/libwrap_syscall.la \
 $(WRAPPERS_LIB)/fork/libwrap_fork.la

#
# Add sampling mechanisms
#
core_MODULES += \
 sampling/common/libsampling_common.la \
 sampling/timer/libsampling_timer.la
if WANT_PEBS_SAMPLING
core_MODULES += sampling/intel-pebs/libsampling_intel_pebs.la
endif
if WANT_JAVA_JVMTI
core_MODULES += $(WRAPPERS_LIB)/JAVA/libwrap_java.la
endif

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
SEQUENTIAL_MERGE_LIB = $(MERGER_LIB)/libmpi2prv.la
if HAVE_LIBADDR2LINE
SEQUENTIAL_MERGE_LIB += $(LIBADDR2LINE_LIBADD)
endif
SEQUENTIAL_CFLAGS_MERGE_LIB = -DEMBED_MERGE_IN_TRACE -I$(MERGER_INC)/common
if HAVE_MPI
PARALLEL_MERGE_LIB = $(MERGER_LIB)/parallel/libmpimpi2prv.la
PARALLEL_MERGE_LIB_AIX = $(MERGER_INC)/parallel/.libs/libmpimpi2prv.a
if HAVE_LIBADDR2LINE
PARALLEL_MERGE_LIB += $(LIBADDR2LINE_LIBADD)
PARALLEL_MERGE_LIB_AIX += $(LIBADDR2LINE_LIBADD)
endif
PARALLEL_CFLAGS_MERGE_LIB = -DEMBED_MERGE_IN_TRACE -I$(MERGER_INC)/common
endif
endif
endif
else
if WANT_MERGE_IN_TRACE
SEQUENTIAL_MERGE_LIB = $(MERGER_LIB)/libmpi2prv.la
if HAVE_LIBADDR2LINE
SEQUENTIAL_MERGE_LIB += $(LIBADDR2LINE_LIBADD)
endif
SEQUENTIAL_CFLAGS_MERGE_LIB = -DEMBED_MERGE_IN_TRACE -I$(MERGER_INC)/common
if HAVE_MPI
PARALLEL_MERGE_LIB = $(MERGER_LIB)/parallel/libmpimpi2prv.la
PARALLEL_MERGE_LIB_AIX = $(MERGER_INC)/parallel/.libs/libmpimpi2prv.a
if HAVE_LIBADDR2LINE
PARALLEL_MERGE_LIB += $(LIBADDR2LINE_LIBADD)
PARALLEL_MERGE_LIB_AIX += $(LIBADDR2LINE_LIBADD)
endif
PARALLEL_CFLAGS_MERGE_LIB = -DEMBED_MERGE_IN_TRACE -I$(MERGER_INC)/common
endif
endif
endif

if HAVE_MPI

if SINGLE_MPI_LIBRARY
ADDITIONAL_MPI_LIBS_CF = $(INTERFACES_LIB)/MPI/libiface_mpicf.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_cf.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la
ADDITIONAL_MPI_BURST_LIBS_CF = $(INTERFACES_LIB)/MPI/libiface_mpicf.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_burst_cf.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la
ADDITIONAL_MPI_DYNLIBS_CF = $(INTERFACES_LIB)/MPI/lib_dyn_iface_mpicf.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_cf.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la
endif

ADDITIONAL_MPI_LIBS_C = $(INTERFACES_LIB)/MPI/libiface_mpi.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_c.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la
ADDITIONAL_MPI_BURST_LIBS_C = $(INTERFACES_LIB)/MPI/libiface_mpi.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_burst_c.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la
ADDITIONAL_MPI_DYNLIBS_C = $(INTERFACES_LIB)/MPI/lib_dyn_iface_mpic.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_c.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la


ADDITIONAL_MPI_LIBS_F = $(INTERFACES_LIB)/MPI/libiface_mpif.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_f.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la
ADDITIONAL_MPI_BURST_LIBS_F = $(INTERFACES_LIB)/MPI/libiface_mpif.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_burst_f.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la
ADDITIONAL_MPI_DYNLIBS_F = $(INTERFACES_LIB)/MPI/lib_dyn_iface_mpif.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_f.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la
endif

ADDITIONAL_GASPI_LIBS = $(WRAPPERS_LIB)/GASPI/libwrap_gaspi.la

ADDITIONAL_PTHREAD_LIBS = $(WRAPPERS_LIB)/pthread/libwrap_pthread.la $(PTHREAD_LIBS)
ADDITIONAL_OPENMP_LIBS = $(WRAPPERS_LIB)/OMP/libwrap_openmp.la
NEW_ADDITIONAL_OPENMP_LIBS = $(WRAPPERS_LIB)/OMP_new/libwrap_openmp.la
NEW_ADDITIONAL_OPENMP_BURST_LIBS = $(WRAPPERS_LIB)/OMP_new/libwrap_openmp_burst.la
ADDITIONAL_OPENCL_LIBS = $(WRAPPERS_LIB)/OPENCL/libwrap_opencl.la
ADDITIONAL_CUDA_LIBS = $(WRAPPERS_LIB)/CUDA/libwrap_cuda.la
ADDITIONAL_OPENSHMEM_LIBS = $(WRAPPERS_LIB)/openshmem/libwrap_openshmem.la
ADDITIONAL_OPENACC_LIBS = $(WRAPPERS_LIB)/OPENACC/libwrap_openacc.la
if HAVE_CUDA
ADDITIONAL_OPENACCCUDA_LIBS = $(WRAPPERS_LIB)/OPENACC/libwrap_openacccuda.la
endif

libmpigaspitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(ADDITIONAL_GASPI_LIBS) $(SEQUENTIAL_MERGE_LIB)
libgaspitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_GASPI_LIBS) $(SEQUENTIAL_MERGE_LIB)
libompigaspitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(ADDITIONAL_GASPI_LIBS) $(SEQUENTIAL_MERGE_LIB) $(ADDITIONAL_OPENMP_LIBS)
libompgaspitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_GASPI_LIBS) $(SEQUENTIAL_MERGE_LIB) $(ADDITIONAL_OPENMP_LIBS)
libptmpigaspitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(ADDITIONAL_MPI_LIBS_C) $(ADDITIONAL_GASPI_LIBS) $(SEQUENTIAL_MERGE_LIB) $(ADDITIONAL_PTHREAD_LIBS)
libptgaspitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(ADDITIONAL_GASPI_LIBS) $(SEQUENTIAL_MERGE_LIB) $(ADDITIONAL_PTHREAD_LIBS)
libcudampigaspitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(ADDITIONAL_GASPI_LIBS) $(SEQUENTIAL_MERGE_LIB) $(ADDITIONAL_CUDA_LIBS)
libcudagaspitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_GASPI_LIBS) $(SEQUENTIAL_MERGE_LIB) $(ADDITIONAL_CUDA_LIBS)
liboclmpigaspitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(ADDITIONAL_GASPI_LIBS) $(SEQUENTIAL_MERGE_LIB) $(ADDITIONAL_OPENCL_LIBS)
liboclgaspitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_GASPI_LIBS) $(SEQUENTIAL_MERGE_LIB) $(ADDITIONAL_OPENCL_LIBS)
libnanosmpigaspitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(ADDITIONAL_GASPI_LIBS) $(SEQUENTIAL_MERGE_LIB)
libnanosgaspitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_GASPI_LIBS) $(SEQUENTIAL_MERGE_LIB)

libcudatrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_CUDA_LIBS) $(SEQUENTIAL_MERGE_LIB)
libcudampitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)
libcudampitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_F) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)
libcudaomptrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_OPENMP_LIBS) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)
libcudanewomptrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(NEW_ADDITIONAL_OPENMP_LIBS) $(ADDITIONAL_CUDA_LIBS) $(SEQUENTIAL_MERGE_LIB)
libptcudatrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(ADDITIONAL_PTHREAD_LIBS) $(ADDITIONAL_CUDA_LIBS) $(SEQUENTIAL_MERGE_LIB)
libptcudampitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(ADDITIONAL_PTHREAD_LIBS) $(ADDITIONAL_MPI_LIBS_C) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)
libptcudampitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(ADDITIONAL_PTHREAD_LIBS) $(ADDITIONAL_MPI_LIBS_F) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libcudampitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_CF) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)
endif

libocltrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_OPENCL_LIBS) $(SEQUENTIAL_MERGE_LIB)
liboclmpitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(ADDITIONAL_OPENCL_LIBS) $(PARALLEL_MERGE_LIB)
liboclmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_F) $(ADDITIONAL_OPENCL_LIBS) $(PARALLEL_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  liboclmpitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_CF) $(ADDITIONAL_OPENCL_LIBS) $(PARALLEL_MERGE_LIB)
endif

liboacctrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_OPENACC_LIBS) $(SEQUENTIAL_MERGE_LIB)
liboaccmpitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_OPENACC_LIBS) $(ADDITIONAL_MPI_LIBS_C) $(PARALLEL_MERGE_LIB)
liboaccmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_OPENACC_LIBS) $(ADDITIONAL_MPI_LIBS_F) $(PARALLEL_MERGE_LIB)
liboacccudatrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_OPENACCCUDA_LIBS) $(ADDITIONAL_CUDA_LIBS) $(SEQUENTIAL_MERGE_LIB)
liboacccudampitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_OPENACCCUDA_LIBS) $(ADDITIONAL_CUDA_LIBS) $(ADDITIONAL_MPI_LIBS_C) $(PARALLEL_MERGE_LIB)
liboacccudampitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_OPENACCCUDA_LIBS) $(ADDITIONAL_CUDA_LIBS) $(ADDITIONAL_MPI_LIBS_F) $(PARALLEL_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  liboaccmpitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_OPENACC_LIBS) $(ADDITIONAL_MPI_LIBS_CF) $(PARALLEL_MERGE_LIB)
  liboacccudampitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_OPENACCCUDA_LIBS) $(ADDITIONAL_CUDA_LIBS) $(ADDITIONAL_MPI_LIBS_CF) $(PARALLEL_MERGE_LIB)
endif

libcudaompitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(ADDITIONAL_OPENMP_LIBS) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)
libcudaompitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_F) $(ADDITIONAL_OPENMP_LIBS) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)
libcudanewompitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(NEW_ADDITIONAL_OPENMP_LIBS) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)
libcudanewompitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_F) $(NEW_ADDITIONAL_OPENMP_LIBS) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libcudaompitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_CF) $(ADDITIONAL_OPENMP_LIBS) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)
  libcudanewompitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_CF) $(NEW_ADDITIONAL_OPENMP_LIBS) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)
endif

libseqtrace_la_LIBADD  = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)
libmpitrace_la_LIBADD  = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_BURST_LIBS_C) $(PARALLEL_MERGE_LIB)
libmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_BURST_LIBS_F) $(PARALLEL_MERGE_LIB)
libptmpitrace_la_LIBADD  = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(ADDITIONAL_PTHREAD_LIBS) $(ADDITIONAL_MPI_LIBS_C) $(PARALLEL_MERGE_LIB)
libptmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(ADDITIONAL_PTHREAD_LIBS) $(ADDITIONAL_MPI_LIBS_F) $(PARALLEL_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libmpitracecf_la_LIBADD  = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_BURST_LIBS_CF) $(PARALLEL_MERGE_LIB)
  libptmpitracecf_la_LIBADD  = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(ADDITIONAL_PTHREAD_LIBS) $(ADDITIONAL_MPI_LIBS_CF) $(PARALLEL_MERGE_LIB)
endif

libompitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(ADDITIONAL_OPENMP_LIBS) $(PARALLEL_MERGE_LIB)
libompitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_F) $(ADDITIONAL_OPENMP_LIBS) $(PARALLEL_MERGE_LIB)
libomptrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_OPENMP_LIBS) $(SEQUENTIAL_MERGE_LIB)
libnewomptrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(NEW_ADDITIONAL_OPENMP_BURST_LIBS) $(SEQUENTIAL_MERGE_LIB)
libnewompitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_BURST_LIBS_C) $(NEW_ADDITIONAL_OPENMP_BURST_LIBS) $(PARALLEL_MERGE_LIB)
libnewompitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_BURST_LIBS_F) $(NEW_ADDITIONAL_OPENMP_BURST_LIBS) $(PARALLEL_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libompitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_CF) $(ADDITIONAL_OPENMP_LIBS) $(PARALLEL_MERGE_LIB)
  libnewompitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_BURST_LIBS_CF) $(NEW_ADDITIONAL_OPENMP_BURST_LIBS) $(PARALLEL_MERGE_LIB)
endif

libsmpssmpitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(PARALLEL_MERGE_LIB)
libsmpssmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_F) $(PARALLEL_MERGE_LIB)
libsmpsstrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libsmpssmpitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_CF) $(PARALLEL_MERGE_LIB)
endif

libnanostrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)
libnanosmpitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(PARALLEL_MERGE_LIB)
libnanosmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_F) $(PARALLEL_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libnanosmpitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_CF) $(PARALLEL_MERGE_LIB)
endif

libpttrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(ADDITIONAL_PTHREAD_LIBS) $(SEQUENTIAL_MERGE_LIB)

libopenshmemtrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_OPENSHMEM_LIBS) $(SEQUENTIAL_MERGE_LIB)

lib_dyn_mpitracec_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_DYNLIBS_C) $(PARALLEL_MERGE_LIB)
lib_dyn_mpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_DYNLIBS_F) $(PARALLEL_MERGE_LIB)
lib_dyn_mpitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_DYNLIBS_CF) $(PARALLEL_MERGE_LIB)
lib_dyn_ompitracec_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_DYNLIBS_C) $(ADDITIONAL_OPENMP_LIBS) $(PARALLEL_MERGE_LIB)
lib_dyn_ompitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_DYNLIBS_F) $(ADDITIONAL_OPENMP_LIBS) $(PARALLEL_MERGE_LIB)
lib_dyn_ompitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_DYNLIBS_CF) $(ADDITIONAL_OPENMP_LIBS) $(PARALLEL_MERGE_LIB)
lib_dyn_cudampitracec_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_DYNLIBS_C) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)
lib_dyn_cudampitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_DYNLIBS_F) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)
lib_dyn_cudampitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_DYNLIBS_CF) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)

#
# Compilation flags
#
core_INCLUDES = \
 -I$(TRACER_INC) -I$(HWC_INC) -I$(CLOCKS_INC) -I$(INTERFACES_INC)/API -I$(INTERFACES_INC)/MPI \
 -I$(BURST_MODE_DIR) \
 -I$(STATS_DIR) \
 -I$(STATS_DIR)/MPI \
 -I$(STATS_DIR)/OMP \
 -I$(WRAPPERS_INC)/API \
 -I$(WRAPPERS_INC)/MPI \
 -I$(WRAPPERS_INC)/OMP \
 -I$(WRAPPERS_INC)/pthread \
 -I$(WRAPPERS_INC)/CUDA \
 -I$(WRAPPERS_INC)/OPENCL \
 -I$(WRAPPERS_INC)/fork \
 -I$(WRAPPERS_INC)/MALLOC \
 -I$(WRAPPERS_INC)/IO \
 -I$(WRAPPERS_INC)/SYSCALL \
 -I$(WRAPPERS_INC)/openshmem \
 -I$(WRAPPERS_INC)/OPENACC \
 -I$(WRAPPERS_INC)/GASPI \
 -I$(SAMPLING_DIR)/common \
 -I$(top_srcdir) -I$(top_srcdir)/include -I$(COMMON_INC) $(UNWIND_CFLAGS)

# XXX Folder OMP_new will be renamed to OMP (already included by core_INCLUDES above) when deprecated OpenMP support is removed
NEW_OMP_INCLUDES = -I$(WRAPPERS_INC)/OMP_new 

burst_CFLAGS  = -DHAVE_BURST

if HAVE_PAPI
if !HAVE_PAPI_EMBEDDED
  core_INCLUDES += @PAPI_CFLAGS@
endif
endif
if HAVE_XML2
  core_INCLUDES += @XML2_CFLAGS@
endif
core_INCLUDES += -I$(SAMPLING_DIR)/timer
if WANT_PEBS_SAMPLING
  core_INCLUDES += -I$(SAMPLING_DIR)/intel-pebs
endif

if HAVE_PAPI
if !HAVE_PAPI_EMBEDDED
  PAPI_LINKER_FLAGS = @PAPI_LDFLAGS@ @PAPI_RPATH@ @PAPI_LIBS@
endif
endif
if HAVE_PMAPI
  PMAPI_LINKER_FLAGS = -lpmapi
endif
HWC_LINKER_FLAGS = $(PAPI_LINKER_FLAGS) $(PMAPI_LINKER_FLAGS)

if HAVE_XML2
  XML2_LINKER_FLAGS = @XML2_LDFLAGS@ @XML2_RPATH@ @XML2_LIBS@
endif
if HAVE_LIBDL
  LDL = -ldl
endif
if HAVE_UNWIND
  UNWIND_LINKER_FLAGS = @UNWIND_LDFLAGS@ @UNWIND_RPATH@ @UNWIND_LIBS@
endif

# POSIX clock may be available by default. Avoid adding these if unneeded.
if USE_POSIX_CLOCK
if NEED_POSIX_CLOCK 
  POSIXCLOCK_LINKER_FLAGS = @RT_LDFLAGS@ @RT_RPATH@ @RT_LIBS@
endif
endif

if HAVE_ZLIB
  LIBZ_LINKER_FLAGS = @LIBZ_LDFLAGS@ @LIBZ_RPATH@ @LIBZ_LIBS@
endif

core_CFLAGS =
if PTHREAD_SUPPORT_IN_ALL_LIBS
core_CFLAGS += -DPTHREAD_SUPPORT
endif
if OS_ANDROID
core_CFLAGS += -DHOST_NAME_MAX=255
endif

libmpigaspitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DMPI_SUPPORT $(MPI_CFLAGS) -DGASPI_SUPPORT $(GASPI_CFLAGS)
libgaspitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DGASPI_SUPPORT $(GASPI_CFLAGS)
libompigaspitrace_la_CFLAGS = $(libmpigaspitrace_la_CFLAGS) -DOMP_SUPPORT
libompgaspitrace_la_CFLAGS = $(libgaspitrace_la_CFLAGS) -DOMP_SUPPORT
libptmpigaspitrace_la_CFLAGS = $(libmpigaspitrace_la_CFLAGS) -DPTHREAD_SUPPORT $(PTHREAD_CFLAGS)
libptgaspitrace_la_CFLAGS = $(libgaspitrace_la_CFLAGS) -DPTHREAD_SUPPORT $(PTHREAD_CFLAGS)
libcudampigaspitrace_la_CFLAGS = $(libmpigaspitrace_la_CFLAGS) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
libcudagaspitrace_la_CFLAGS = $(libgaspitrace_la_CFLAGS) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
liboclmpigaspitrace_la_CFLAGS = $(libmpigaspitrace_la_CFLAGS) -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS)
liboclgaspitrace_la_CFLAGS = $(libgaspitrace_la_CFLAGS) -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS)
libnanosmpigaspitrace_la_CFLAGS = $(libmpigaspitrace_la_CFLAGS) -DNANOS_SUPPORT
libnanosgaspitrace_la_CFLAGS = $(libgaspitrace_la_CFLAGS) -DNANOS_SUPPORT

libcudatrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
libcudampitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
libcudampitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
libcudaomptrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DOMP_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
libcudanewomptrace_la_CFLAGS = $(NEW_OMP_INCLUDES) $(core_INCLUDES) $(core_CFLAGS) -DNEW_OMP_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
libptcudatrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT -DPTHREAD_SUPPORT $(WRAPPERS_CUDA_CFLAGS) $(PTHREAD_CFLAGS)
libptcudampitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT -DPTHREAD_SUPPORT $(WRAPPERS_CUDA_CFLAGS) $(PTHREAD_CFLAGS)
libptcudampitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT -DPTHREAD_SUPPORT $(WRAPPERS_CUDA_CFLAGS) $(PTHREAD_CFLAGS)

if SINGLE_MPI_LIBRARY
  libcudampitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
endif

libocltrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS)
liboclmpitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS)
liboclmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS)

if SINGLE_MPI_LIBRARY
  liboclmpitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS)
endif

liboacctrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DOPENACC_SUPPORT $(WRAPPERS_OPENACC_CFLAGS)
liboaccmpitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DOPENACC_SUPPORT $(WRAPPERS_OPENACC_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
liboaccmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DOPENACC_SUPPORT $(WRAPPERS_OPENACC_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
liboacccudatrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DOPENACC_SUPPORT $(WRAPPERS_OPENACC_CFLAGS) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
liboacccudampitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DOPENACC_SUPPORT $(WRAPPERS_OPENACC_CFLAGS) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
liboacccudampitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DOPENACC_SUPPORT $(WRAPPERS_OPENACC_CFLAGS) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  liboaccmpitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DOPENACC_SUPPORT $(WRAPPERS_OPENACC_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
  liboacccudampitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DOPENACC_SUPPORT $(WRAPPERS_OPENACC_CFLAGS) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
endif

libcudaompitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS) -DOMP_SUPPORT
libcudaompitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS) -DOMP_SUPPORT

if SINGLE_MPI_LIBRARY
  libcudaompitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS) -DOMP_SUPPORT
  libcudanewompitracecf_la_CFLAGS = $(NEW_OMP_INCLUDES) $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS) -DNEW_OMP_SUPPORT
endif

libseqtrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB)

libmpitrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) $(burst_CFLAGS)
libmpitracef_la_CFLAGS = $(libmpitrace_la_CFLAGS)

if SINGLE_MPI_LIBRARY
  libmpitracecf_la_CFLAGS  = $(libmpitracef_la_CFLAGS)
endif

libptmpitrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DPTHREAD_SUPPORT $(PTHREAD_CFLAGS)
libptmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DPTHREAD_SUPPORT $(PTHREAD_CFLAGS)

if SINGLE_MPI_LIBRARY
  libptmpitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DPTHREAD_SUPPORT $(PTHREAD_CFLAGS)
endif

libompitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DOMP_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
libompitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DOMP_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
libomptrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DOMP_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)
libnewomptrace_la_CFLAGS = $(NEW_OMP_INCLUDES) $(core_INCLUDES) $(core_CFLAGS) -DNEW_OMP_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB) $(burst_CFLAGS)
libnewompitrace_la_CFLAGS = $(NEW_OMP_INCLUDES) $(core_INCLUDES) $(core_CFLAGS) -DNEW_OMP_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) $(burst_CFLAGS)
libnewompitracef_la_CFLAGS = $(libnewompitrace_la_CFLAGS)
libcudanewompitrace_la_CFLAGS = $(NEW_OMP_INCLUDES) $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS) -DNEW_OMP_SUPPORT
libcudanewompitracef_la_CFLAGS = $(NEW_OMP_INCLUDES) $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS) -DNEW_OMP_SUPPORT

if SINGLE_MPI_LIBRARY
  libompitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DOMP_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
  libnewompitracecf_la_CFLAGS = $(libnewompitracef_la_CFLAGS)
endif

libsmpssmpitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DSMPSS_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
libsmpssmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DSMPSS_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
libsmpsstrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DSMPSS_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libsmpssmpitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DSMPSS_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
endif

libnanostrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DNANOS_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)
libnanosmpitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DNANOS_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
libnanosmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DNANOS_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libnanosmpitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DNANOS_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
endif

libpttrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DPTHREAD_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB) $(PTHREAD_CFLAGS)

libopenshmemtrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @OPENSHMEM_CFLAGS@ -DOPENSHMEM_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)

lib_dyn_mpitracec_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DDYNINST_MODULE
lib_dyn_mpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DDYNINST_MODULE
lib_dyn_mpitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DDYNINST_MODULE
lib_dyn_ompitracec_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DDYNINST_MODULE -DOMP_SUPPORT
lib_dyn_ompitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DDYNINST_MODULE -DOMP_SUPPORT
lib_dyn_ompitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DDYNINST_MODULE -DOMP_SUPPORT
lib_dyn_cudampitracec_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DDYNINST_MODULE -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
lib_dyn_cudampitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DDYNINST_MODULE -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
lib_dyn_cudampitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DDYNINST_MODULE -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
lib_dyn_oclmpitracec_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DDYNINST_MODULE -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS)
lib_dyn_oclmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DDYNINST_MODULE -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS)
lib_dyn_oclmpitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DDYNINST_MODULE -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS)

#
# Online support is only activated for the MPI libraries
# 
if HAVE_ONLINE
libmpitrace_la_LIBADD        += $(ONLINE_LIB)/libonline.la
libmpitracef_la_LIBADD       += $(ONLINE_LIB)/libonline.la
libompitrace_la_LIBADD       += $(ONLINE_LIB)/libonline.la
libompitracef_la_LIBADD      += $(ONLINE_LIB)/libonline.la
lib_dyn_mpitracec_la_LIBADD  += $(ONLINE_LIB)/libonline.la
lib_dyn_mpitracef_la_LIBADD  += $(ONLINE_LIB)/libonline.la
lib_dyn_mpitracecf_la_LIBADD += $(ONLINE_LIB)/libonline.la

libmpitrace_la_CFLAGS        += -DHAVE_ONLINE -I$(ONLINE_INC)
libmpitracef_la_CFLAGS       += -DHAVE_ONLINE -I$(ONLINE_INC)
libompitrace_la_CFLAGS       += -DHAVE_ONLINE -I$(ONLINE_INC)
libompitracef_la_CFLAGS      += -DHAVE_ONLINE -I$(ONLINE_INC)
lib_dyn_mpitracec_la_CFLAGS  += -DHAVE_ONLINE -I$(ONLINE_INC)
lib_dyn_mpitracef_la_CFLAGS  += -DHAVE_ONLINE -I$(ONLINE_INC)
lib_dyn_mpitracecf_la_CFLAGS += -DHAVE_ONLINE -I$(ONLINE_INC)

if SINGLE_MPI_LIBRARY
  libmpitracecf_la_LIBADD    += $(ONLINE_LIB)/libonline.la
  libompitracecf_la_LIBADD   += $(ONLINE_LIB)/libonline.la

  libmpitracecf_la_CFLAGS    += -DHAVE_ONLINE -I$(ONLINE_INC)
  libompitracecf_la_CFLAGS   += -DHAVE_ONLINE -I$(ONLINE_INC)
endif

endif

#
# Link flags
#
if HAVE_MPI_WITH_SHARED_LIBS
NEW_MPI_LIBS = @MPI_LDFLAGS@ @MPI_RPATH@ @MPI_LIBS@
else
NEW_MPI_LIBS = @MPI_LDFLAGS@ @MPI_LIBS@ -static
endif

COMMON_LINKER_FLAGS = \
 -release $(PACKAGE_VERSION) \
 $(HWC_LINKER_FLAGS) \
 $(UNWIND_LINKER_FLAGS) \
 $(POSIXCLOCK_LINKER_FLAGS) \
 $(XML2_LINKER_FLAGS) \
 $(LDL)\
 $(LIBZ_LINKER_FLAGS) \
 $(MRNET_LINKER_FLAGS)

libmpigaspitrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libgaspitrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libompigaspitrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libompgaspitrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libptmpigaspitrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS) $(PTHREAD_LIBS)
libptgaspitrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS) $(PTHREAD_LIBS)
libcudampigaspitrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libcudagaspitrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
liboclmpigaspitrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
liboclgaspitrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libnanosmpigaspitrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libnanosgaspitrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)

libseqtrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)

libmpitrace_la_LDFLAGS = $(NEW_MPI_LIBS) $(COMMON_LINKER_FLAGS)
libmpitracef_la_LDFLAGS = $(libmpitrace_la_LDFLAGS) # -static 

if SINGLE_MPI_LIBRARY
  libmpitracecf_la_LDFLAGS = $(libmpitrace_la_LDFLAGS) # -static 
endif

libptmpitrace_la_LDFLAGS = $(NEW_MPI_LIBS) $(COMMON_LINKER_FLAGS) $(PTHREAD_LIBS)
libptmpitracef_la_LDFLAGS = $(libptmpitrace_la_LDFLAGS) # -static 

if SINGLE_MPI_LIBRARY
  libptmpitracecf_la_LDFLAGS = $(libptmpitrace_la_LDFLAGS) # -static 
endif

liboacctrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
liboaccmpitrace_la_LDFLAGS = $(NEW_MPI_LIBS) $(liboacctrace_la_LDFLAGS)
liboaccmpitracef_la_LDFLAGS = $(NEW_MPI_LIBS) $(liboacctrace_la_LDFLAGS)
liboacccudatrace_la_LDFLAGS = $(liboacctrace_la_LDFLAGS) @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@
liboacccudampitrace_la_LDFLAGS = $(NEW_MPI_LIBS) $(liboacctrace_la_LDFLAGS) @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@
liboacccudampitracef_la_LDFLAGS = $(NEW_MPI_LIBS) $(liboacctrace_la_LDFLAGS) @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@

if SINGLE_MPI_LIBRARY
  liboaccmpitracecf_la_LDFLAGS = $(NEW_MPI_LIBS) $(liboacctrace_la_LDFLAGS)
  liboacccudampitracecf_la_LDFLAGS = $(NEW_MPI_LIBS) $(liboacctrace_la_LDFLAGS) @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@
endif

libcudatrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS) @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@
libcudampitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS) @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@
libcudampitracef_la_LDFLAGS = $(libmpitracef_la_LDFLAGS) @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@
libcudaomptrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS) @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@
libcudanewomptrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS) -dynamic @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@
libptcudatrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS) $(PTHREAD_LIBS) @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@
libptcudampitrace_la_LDFLAGS = $(libptmpitrace_la_LDFLAGS) $(PTHREAD_LIBS) @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@
libptcudampitracef_la_LDFLAGS = $(libptmpitracef_la_LDFLAGS) $(PTHREAD_LIBS) @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@

if SINGLE_MPI_LIBRARY
  libcudampitracecf_la_LDFLAGS = $(libmpitracecf_la_LDFLAGS) @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@
endif

libcudaompitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS) @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@
libcudaompitracef_la_LDFLAGS = $(libmpitracef_la_LDFLAGS) @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@
libcudanewompitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS) -dynamic @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@
libcudanewompitracef_la_LDFLAGS = $(libmpitracef_la_LDFLAGS) -dynamic @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@

if SINGLE_MPI_LIBRARY
  libcudaompitracecf_la_LDFLAGS = $(libmpitracecf_la_LDFLAGS) @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@
  libcudanewompitracecf_la_LDFLAGS = $(libmpitracecf_la_LDFLAGS) -dynamic @CUDA_LDFLAGS_RPATH@ @CUPTI_LDFLAGS_RPATH@ @CUDA_LIBS@ @CUPTI_LIBS@
endif

libocltrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS) @OPENCL_LDFLAGS_RPATH@ @OPENCL_LIBS@
liboclmpitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS) @OPENCL_LDFLAGS_RPATH@ @OPENCL_LIBS@
liboclmpitracef_la_LDFLAGS = $(libmpitracef_la_LDFLAGS) @OPENCL_LDFLAGS_RPATH@ @OPENCL_LIBS@

if SINGLE_MPI_LIBRARY
  liboclmpitracecf_la_LDFLAGS = $(libmpitracecf_la_LDFLAGS) @OPENCL_LDFLAGS_RPATH@ @OPENCL_LIBS@
endif

libomptrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libnewomptrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS) -dynamic
libompitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libompitracef_la_LDFLAGS = $(libmpitracef_la_LDFLAGS)
libnewompitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS) -dynamic
libnewompitracef_la_LDFLAGS = $(libmpitrace_la_LDFLAGS) -dynamic

if SINGLE_MPI_LIBRARY
  libompitracecf_la_LDFLAGS = $(libmpitracecf_la_LDFLAGS)
  libnewompitracecf_la_LDFLAGS = $(libmpitrace_la_LDFLAGS) -dynamic
endif

libsmpsstrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libsmpssmpitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libsmpssmpitracef_la_LDFLAGS = $(libmpitracef_la_LDFLAGS)

if SINGLE_MPI_LIBRARY
  libsmpssmpitracecf_la_LDFLAGS = $(libmpitracecf_la_LDFLAGS)
endif

libnanostrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libnanosmpitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libnanosmpitracef_la_LDFLAGS = $(libmpitracef_la_LDFLAGS)

if SINGLE_MPI_LIBRARY
  libnanosmpitracecf_la_LDFLAGS = $(libmpitracecf_la_LDFLAGS)
endif

libpttrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS) $(PTHREAD_LIBS)

libopenshmemtrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS) $(OPENSHMEM_LDFLAGS)

lib_dyn_mpitracec_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
lib_dyn_mpitracef_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
lib_dyn_mpitracecf_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
lib_dyn_ompitracec_la_LDFLAGS = $(lib_dyn_mpitracec_la_LDFLAGS) 
lib_dyn_ompitracef_la_LDFLAGS = $(lib_dyn_mpitracef_la_LDFLAGS) 
lib_dyn_ompitracecf_la_LDFLAGS = $(lib_dyn_mpitracecf_la_LDFLAGS) 
lib_dyn_cudampitracec_la_LDFLAGS = $(lib_dyn_mpitracec_la_LDFLAGS) 
lib_dyn_cudampitracef_la_LDFLAGS = $(lib_dyn_mpitracef_la_LDFLAGS) 
lib_dyn_cudampitracecf_la_LDFLAGS = $(lib_dyn_mpitracecf_la_LDFLAGS) 

## -static tells libtool to generate ONLY the static library (we don't want
##   the libmpitracef.so because it don't preloads fine with programs)
## -shared tells libtool to generate ONLY shared libraries (right now,
##   only for dyninst modules)
## -R flag works like -Wl,-rpath
## -no-undefined could be interesting for AIX?
##   check http://www.andamooka.org/reader.pl?pgid=autobookautobook_89
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
#libextrae_mpi_la_LIBADD  = $(INTERFACES_LIB)/MPI/libiface_mpi.la $(PARALLEL_MERGE_LIB)
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

