include $(top_srcdir)/PATHS

SUBDIRS = clocks hwc interfaces wrappers stats sampling

if HAVE_ONLINE
  SUBDIRS += online
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

if WANT_CUDAorCUPTI
  lib_LTLIBRARIES += libcudatrace.la
endif

if WANT_OPENCL
  lib_LTLIBRARIES += libocltrace.la
endif

if HAVE_MPI
  lib_LTLIBRARIES += libmpitrace.la libmpitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += libmpitracecf.la
endif
if WANT_OPENMP
  lib_LTLIBRARIES += libompitrace.la libompitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += libompitracecf.la
endif
endif
if WANT_CUDAorCUPTI
  lib_LTLIBRARIES += libcudampitrace.la libcudampitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += libcudampitracecf.la
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

if HAVE_MPI
if WANT_OPENMP
if WANT_CUDAorCUPTI
  lib_LTLIBRARIES += libcudaompitrace.la libcudaompitracef.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += libcudaompitracecf.la
endif
endif
endif
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

if GENERATE_LOAD_BALANCING
if HAVE_MPI
  lib_LTLIBRARIES += libmpitrace-lb.la libmpitracef-lb.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += libmpitracecf-lb.la
endif
if WANT_OPENMP
  lib_LTLIBRARIES += libompitrace-lb.la libompitracef-lb.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += libompitracecf-lb.la
endif
endif
if WANT_SMPSS
  lib_LTLIBRARIES += libsmpssmpitrace-lb.la libsmpssmpitracef-lb.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += libsmpssmpitracecf-lb.la
endif
endif
if WANT_NANOS
  lib_LTLIBRARIES += libnanosmpitrace-lb.la libnanosmpitracef-lb.la
if SINGLE_MPI_LIBRARY
  lib_LTLIBRARIES += libnanosmpitracecf-lb.la
endif
endif
endif
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
if WANT_CUDAorCUPTI
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

WRAPPERS_CORE = \
 wrappers/API/buffers.c wrappers/API/buffers.h \
 wrappers/API/wrapper.c wrappers/API/wrapper.h \
 wrappers/API/misc_wrapper.c wrappers/API/misc_wrapper.h \
 wrappers/API/threadinfo.c wrappers/API/threadinfo.h \
 wrappers/API/trace_buffers.h \
 trace_macros.h trace_macros_mpi.h trace_macros_omp.h \
 trace_hwc.h

core_SRCS = \
 calltrace.c calltrace.h                     \
 signals.c signals.h                         \
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
libcudampitrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_CUDA) $(auto_fini_instr_SRCS)
libcudampitracef_la_SOURCES = $(libcudampitrace_la_SOURCES) extrae_mpif.h

if SINGLE_MPI_LIBRARY
  libcudampitracecf_la_SOURCES = $(libcudampitrace_la_SOURCES) extrae_mpif.h
endif

# OpenCL trace
libocltrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_OCL) $(auto_init_instr_SRCS) $(auto_fini_instr_SRCS)
liboclmpitrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_OCL) $(auto_fini_instr_SRCS)
liboclmpitracef_la_SOURCES = $(liboclmpitrace_la_SOURCES) extrae_mpif.h

if SINGLE_MPI_LIBRARY
  liboclmpitracecf_la_SOURCES = $(liboclmpitrace_la_SOURCES) extrae_mpif.h
endif

# CUDA trace & MPItrace & OpenMP
libcudaompitrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_CUDA)
libcudaompitracef_la_SOURCES = $(libcudaompitrace_la_SOURCES) extrae_mpif.h

if SINGLE_MPI_LIBRARY
  libcudaompitracecf_la_SOURCES = $(libcudaompitrace_la_SOURCES) extrae_mpif.h
endif

# MPItrace & Sequential trace
libmpitrace_la_SOURCES = $(core_SRCS) $(auto_fini_instr_SRCS)
libmpitracef_la_SOURCES = $(libmpitrace_la_SOURCES) extrae_mpif.h
libseqtrace_la_SOURCES = $(core_SRCS) $(auto_init_instr_SRCS) $(auto_fini_instr_SRCS)

if SINGLE_MPI_LIBRARY
  libmpitracecf_la_SOURCES = $(libmpitrace_la_SOURCES) extrae_mpif.h
endif

libptmpitrace_la_SOURCES = $(core_SRCS) $(auto_fini_instr_SRCS)
libptmpitracef_la_SOURCES = $(libptmpitrace_la_SOURCES) extrae_mpif.h

if SINGLE_MPI_LIBRARY
  libptmpitracecf_la_SOURCES = $(libptmpitrace_la_SOURCES) extrae_mpif.h
endif

# OpenMP & OpenMP + MPI
libompitrace_la_SOURCES = $(libmpitrace_la_SOURCES)
libompitracef_la_SOURCES = $(libmpitracef_la_SOURCES)
libomptrace_la_SOURCES = $(core_SRCS) $(auto_init_instr_SRCS) $(auto_fini_instr_SRCS)

if SINGLE_MPI_LIBRARY
  libompitracecf_la_SOURCES = $(libmpitracecf_la_SOURCES)
endif

# SMPSS & SMPSS + MPI
libsmpsstrace_la_SOURCES = $(core_SRCS) $(auto_fini_instr_SRCS)
libsmpssmpitrace_la_SOURCES = $(libmpitrace_la_SOURCES)
libsmpssmpitracef_la_SOURCES = $(libmpitracef_la_SOURCES)

if SINGLE_MPI_LIBRARY
  libsmpssmpitracecf_la_SOURCES = $(libmpitracecf_la_SOURCES)
endif

# Nanos
libnanostrace_la_SOURCES = $(core_SRCS)
libnanosmpitrace_la_SOURCES = $(core_SRCS)
libnanosmpitracef_la_SOURCES = $(libnanosmpitrace_la_SOURCES) extrae_mpif.h

if SINGLE_MPI_LIBRARY
  libnanosmpitracecf_la_SOURCES = $(libnanosmpitrace_la_SOURCES) extrae_mpif.h
endif

# pthreads
libpttrace_la_SOURCES = $(core_SRCS) $(auto_init_instr_SRCS) $(auto_fini_instr_SRCS)

# OpenSHMEM
libopenshmemtrace_la_SOURCES = $(core_SRCS) $(auto_fini_instr_SRCS)

# LoadBalancing
libmpitrace_lb_la_SOURCES = $(libmpitrace_la_SOURCES)
libmpitracef_lb_la_SOURCES = $(libmpitracef_la_SOURCES)
libompitrace_lb_la_SOURCES = $(libompitrace_la_SOURCES)
libompitracef_lb_la_SOURCES = $(libompitracef_la_SOURCES)
libsmpssmpitrace_lb_la_SOURCES = $(libmpitrace_la_SOURCES)
libsmpssmpitracef_lb_la_SOURCES = $(libmpitracef_la_SOURCES)
libnanosmpitrace_lb_la_SOURCES = $(libnanosmpitrace_la_SOURCES)
libnanosmpitracef_lb_la_SOURCES = $(libnanosmpitracef_la_SOURCES)

if SINGLE_MPI_LIBRARY
  libmpitracecf_lb_la_SOURCES = $(libmpitracecf_la_SOURCES)
  libompitracecf_lb_la_SOURCES = $(libompitracecf_la_SOURCES)
  libsmpssmpitracecf_lb_la_SOURCES = $(libmpitracecf_la_SOURCES)
  libnanosmpitracecf_lb_la_SOURCES = $(libnanosmpitracecf_la_SOURCES)
endif

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
if HAVE_BINUTILS
SEQUENTIAL_MERGE_LIB += $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS)
endif
SEQUENTIAL_CFLAGS_MERGE_LIB = -DEMBED_MERGE_IN_TRACE -I$(MERGER_INC)/common
if HAVE_MPI
PARALLEL_MERGE_LIB = $(MERGER_LIB)/parallel/libmpimpi2prv.la
PARALLEL_MERGE_LIB_AIX = $(MERGER_INC)/parallel/.libs/libmpimpi2prv.a
if HAVE_BINUTILS
PARALLEL_MERGE_LIB += $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS)
PARALLEL_MERGE_LIB_AIX += $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS)
endif
PARALLEL_CFLAGS_MERGE_LIB = -DEMBED_MERGE_IN_TRACE -I$(MERGER_INC)/common
endif
endif
endif
else
if WANT_MERGE_IN_TRACE
SEQUENTIAL_MERGE_LIB = $(MERGER_LIB)/libmpi2prv.la
if HAVE_BINUTILS
SEQUENTIAL_MERGE_LIB += $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS)
endif
SEQUENTIAL_CFLAGS_MERGE_LIB = -DEMBED_MERGE_IN_TRACE -I$(MERGER_INC)/common
if HAVE_MPI
PARALLEL_MERGE_LIB = $(MERGER_LIB)/parallel/libmpimpi2prv.la
PARALLEL_MERGE_LIB_AIX = $(MERGER_INC)/parallel/.libs/libmpimpi2prv.a
if HAVE_BINUTILS
PARALLEL_MERGE_LIB += $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS)
PARALLEL_MERGE_LIB_AIX += $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS)
endif
PARALLEL_CFLAGS_MERGE_LIB = -DEMBED_MERGE_IN_TRACE -I$(MERGER_INC)/common
endif
endif
endif

if HAVE_MPI

if SINGLE_MPI_LIBRARY
ADDITIONAL_MPI_LIBS_CF = $(INTERFACES_LIB)/MPI/libiface_mpicf.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_cf.la \
 $(STATS_LIB)/libmpistats.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la
ADDITIONAL_MPI_LIBS_CF_LB = $(INTERFACES_LIB)/MPI/libiface_mpicf_lb.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_cf.la \
 $(STATS_LIB)/libmpistats.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la
ADDITIONAL_MPI_DYNLIBS_CF = $(INTERFACES_LIB)/MPI/lib_dyn_iface_mpicf.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_cf.la \
 $(STATS_LIB)/libmpistats.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la
endif

ADDITIONAL_MPI_LIBS_C = $(INTERFACES_LIB)/MPI/libiface_mpi.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_c.la \
 $(STATS_LIB)/libmpistats.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la
ADDITIONAL_MPI_LIBS_C_LB = $(INTERFACES_LIB)/MPI/libiface_mpi_lb.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_c.la \
 $(STATS_LIB)/libmpistats.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la
ADDITIONAL_MPI_DYNLIBS_C = $(INTERFACES_LIB)/MPI/lib_dyn_iface_mpic.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_c.la \
 $(STATS_LIB)/libmpistats.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la


ADDITIONAL_MPI_LIBS_F = $(INTERFACES_LIB)/MPI/libiface_mpif.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_f.la \
 $(STATS_LIB)/libmpistats.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la
ADDITIONAL_MPI_LIBS_F_LB = $(INTERFACES_LIB)/MPI/libiface_mpif_lb.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_f.la \
 $(STATS_LIB)/libmpistats.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la
ADDITIONAL_MPI_DYNLIBS_F = $(INTERFACES_LIB)/MPI/lib_dyn_iface_mpif.la \
 $(WRAPPERS_LIB)/MPI/libwrap_mpi_f.la \
 $(STATS_LIB)/libmpistats.la \
 $(COMMON_MPI_LIB)/libcommon_mpi.la
endif

ADDITIONAL_PTHREAD_LIBS = $(WRAPPERS_LIB)/pthread/libwrap_pthread.la $(PTHREAD_LIBS)
ADDITIONAL_OPENMP_LIBS = $(WRAPPERS_LIB)/OMP/libwrap_openmp.la
ADDITIONAL_OPENCL_LIBS = $(WRAPPERS_LIB)/OPENCL/libwrap_opencl.la
ADDITIONAL_CUDA_LIBS = $(WRAPPERS_LIB)/CUDA/libwrap_cuda.la
ADDITIONAL_OPENSHMEM_LIBS = $(WRAPPERS_LIB)/openshmem/libwrap_openshmem.la

libcudatrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_CUDA_LIBS) $(SEQUENTIAL_MERGE_LIB)
libcudampitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)
libcudampitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_F) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libcudampitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_CF) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)
endif

libocltrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_OPENCL_LIBS) $(SEQUENTIAL_MERGE_LIB)
liboclmpitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(ADDITIONAL_OPENCL_LIBS) $(PARALLEL_MERGE_LIB)
liboclmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_F) $(ADDITIONAL_OPENCL_LIBS) $(PARALLEL_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  liboclmpitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_CF) $(ADDITIONAL_OPENCL_LIBS) $(PARALLEL_MERGE_LIB)
endif

libcudaompitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(ADDITIONAL_OPENMP_LIBS) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)
libcudaompitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_F) $(ADDITIONAL_OPENMP_LIBS) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libcudaompitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_CF) $(ADDITIONAL_OPENMP_LIBS) $(ADDITIONAL_CUDA_LIBS) $(PARALLEL_MERGE_LIB)
endif

libseqtrace_la_LIBADD  = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)
libmpitrace_la_LIBADD  = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(PARALLEL_MERGE_LIB)
libmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_F) $(PARALLEL_MERGE_LIB)
libptmpitrace_la_LIBADD  = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(ADDITIONAL_PTHREAD_LIBS) $(ADDITIONAL_MPI_LIBS_C) $(PARALLEL_MERGE_LIB)
libptmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(ADDITIONAL_PTHREAD_LIBS) $(ADDITIONAL_MPI_LIBS_F) $(PARALLEL_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libmpitracecf_la_LIBADD  = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_CF) $(PARALLEL_MERGE_LIB)
  libptmpitracecf_la_LIBADD  = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(ADDITIONAL_PTHREAD_LIBS) $(ADDITIONAL_MPI_LIBS_CF) $(PARALLEL_MERGE_LIB)
endif

libompitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C) $(ADDITIONAL_OPENMP_LIBS) $(PARALLEL_MERGE_LIB)
libompitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_F) $(ADDITIONAL_OPENMP_LIBS) $(PARALLEL_MERGE_LIB)
libomptrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_OPENMP_LIBS) $(SEQUENTIAL_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libompitracecf_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_CF) $(ADDITIONAL_OPENMP_LIBS) $(PARALLEL_MERGE_LIB)
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

libmpitrace_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C_LB) $(PARALLEL_MERGE_LIB)
libmpitracef_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_F_LB) $(PARALLEL_MERGE_LIB)
libompitrace_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C_LB) $(ADDITIONAL_OPENMP_LIBS) $(PARALLEL_MERGE_LIB)
libompitracef_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_F_LB) $(ADDITIONAL_OPENMP_LIBS) $(PARALLEL_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libmpitracecf_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_CF_LB) $(PARALLEL_MERGE_LIB)
  libompitracecf_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_CF_LB) $(ADDITIONAL_OPENMP_LIBS) $(PARALLEL_MERGE_LIB)
endif

libsmpssmpitrace_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C_LB) $(PARALLEL_MERGE_LIB)
libsmpssmpitracef_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_F_LB) $(PARALLEL_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libsmpssmpitracecf_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_CF_LB) $(PARALLEL_MERGE_LIB)
endif

libnanosmpitrace_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_C_LB) $(PARALLEL_MERGE_LIB)
libnanosmpitracef_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_F_LB) $(PARALLEL_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libnanosmpitracecf_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(ADDITIONAL_MPI_LIBS_CF_LB) $(PARALLEL_MERGE_LIB)
endif

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
 -I$(STATS_INC)/MPI \
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
 -I$(SAMPLING_DIR)/common \
 -I$(top_srcdir) -I$(top_srcdir)/include -I$(COMMON_INC) $(UNWIND_CFLAGS)

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

libcudatrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
libcudampitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
libcudampitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)

if SINGLE_MPI_LIBRARY
  libcudampitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS)
endif

libocltrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS)
liboclmpitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS)
liboclmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS)

if SINGLE_MPI_LIBRARY
  liboclmpitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DOPENCL_SUPPORT $(WRAPPERS_OCL_CFLAGS)
endif

libcudaompitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS) -DOMP_SUPPORT
libcudaompitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS) -DOMP_SUPPORT

if SINGLE_MPI_LIBRARY
  libcudaompitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT $(WRAPPERS_CUDA_CFLAGS) -DOMP_SUPPORT
endif

libseqtrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB)

libmpitrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
libmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libmpitracecf_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
endif

libptmpitrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DPTHREAD_SUPPORT $(PTHREAD_CFLAGS)
libptmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DPTHREAD_SUPPORT $(PTHREAD_CFLAGS)

if SINGLE_MPI_LIBRARY
  libptmpitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB) -DPTHREAD_SUPPORT $(PTHREAD_CFLAGS)
endif

libompitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DOMP_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
libompitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DOMP_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
libomptrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DOMP_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libompitracecf_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DOMP_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
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

libmpitrace_lb_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
libmpitracef_lb_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
libompitrace_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DOMP_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
libompitracef_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DOMP_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libmpitracecf_lb_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
  libompitracecf_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DOMP_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
endif

libsmpssmpitrace_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DSMPSS_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
libsmpssmpitracef_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DSMPSS_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libsmpssmpitracecf_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DSMPSS_SUPPORT -DMPI_SUPPORT $(MPI_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
endif

libnanosmpitrace_lb_la_CFLAGS = $(libnanosmpitrace_la_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
libnanosmpitracef_lb_la_CFLAGS = $(libnanosmpitracef_la_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)

if SINGLE_MPI_LIBRARY
  libnanosmpitracecf_lb_la_CFLAGS = $(libnanosmpitracecf_la_CFLAGS) $(PARALLEL_CFLAGS_MERGE_LIB)
endif

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
NEW_MPI_LIBS = -L@MPI_SHAREDLIBSDIR@ -R @MPI_SHAREDLIBSDIR@ @MPI_LIBS@
else
NEW_MPI_LIBS = -L@MPI_LIBSDIR@ @MPI_LIBS@ -static
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

libcudaompitrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libcudaompitracef_la_LDFLAGS = $(COMMON_LINKER_FLAGS)

if SINGLE_MPI_LIBRARY
  libcudaompitracecf_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
endif

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

libcudatrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libcudampitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libcudampitracef_la_LDFLAGS = $(libmpitracef_la_LDFLAGS)

if SINGLE_MPI_LIBRARY
  libcudampitracecf_la_LDFLAGS = $(libmpitracecf_la_LDFLAGS)
endif

libocltrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
liboclmpitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
liboclmpitracef_la_LDFLAGS = $(libmpitracef_la_LDFLAGS)

if SINGLE_MPI_LIBRARY
  liboclmpitracecf_la_LDFLAGS = $(libmpitracecf_la_LDFLAGS)
endif

libomptrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)
libompitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libompitracef_la_LDFLAGS = $(libmpitracef_la_LDFLAGS)

if SINGLE_MPI_LIBRARY
  libompitracecf_la_LDFLAGS = $(libmpitracecf_la_LDFLAGS)
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

libmpitrace_lb_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libmpitracef_lb_la_LDFLAGS = $(libmpitracef_la_LDFLAGS) # -static
libompitrace_lb_la_LDFLAGS = $(libompitrace_la_LDFLAGS)
libompitracef_lb_la_LDFLAGS = $(libompitracef_la_LDFLAGS)

if SINGLE_MPI_LIBRARY
  libmpitracecf_lb_la_LDFLAGS = $(libmpitracecf_la_LDFLAGS) # -static
  libompitracecf_lb_la_LDFLAGS = $(libompitracecf_la_LDFLAGS)
endif

libsmpssmpitrace_lb_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libsmpssmpitracef_lb_la_LDFLAGS = $(libmpitrace_lb_la_LDFLAGS) # -static

if SINGLE_MPI_LIBRARY
  libsmpssmpitracecf_lb_la_LDFLAGS = $(libmpitrace_lb_la_LDFLAGS) # -static
endif

libnanosmpitrace_lb_la_LDFLAGS = $(libnanosmpitrace_la_LDFLAGS)
libnanosmpitracef_lb_la_LDFLAGS = $(libnanosmpitracef_la_LDFLAGS)

if SINGLE_MPI_LIBRARY
  libnanosmpitracecf_lb_la_LDFLAGS = $(libnanosmpitracecf_la_LDFLAGS)
endif

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

