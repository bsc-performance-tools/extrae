include $(top_srcdir)/PATHS

SUBDIRS = clocks hwc interfaces

# Include this files/directories for the package generation
EXTRA_DIST = external

if HAVE_MRNET
  SUBDIRS += mrnet
endif

if IS_CELL_MACHINE
  SUBDIRS += spu
endif

#
# Libraries ending with c/f are for C/Fortran libraries. If not specified, 
# they should contain symbols for both languages.
#

lib_LTLIBRARIES = libseqtrace.la

if WANT_TRT
  lib_LTLIBRARIES += libtrttrace.la
endif

if WANT_CUDA
  lib_LTLIBRARIES += libcudatrace.la
endif

if HAVE_PACX
  lib_LTLIBRARIES += libpacxtrace.la libpacxtracef.la
endif

if HAVE_MPI
  lib_LTLIBRARIES += libmpitrace.la libmpitracef.la
if WANT_OPENMP
  lib_LTLIBRARIES += libompitrace.la libompitracef.la
endif
if WANT_SMPSS
  lib_LTLIBRARIES += libsmpssmpitrace.la libsmpssmpitracef.la
endif
if WANT_NANOS
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
endif
if WANT_PTHREAD
  lib_LTLIBRARIES += libpttrace.la
if HAVE_MPI
  lib_LTLIBRARIES += libptmpitrace.la libptmpitracef.la
endif
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
endif
endif
endif

if HAVE_DYNINST
if HAVE_MPI
  lib_LTLIBRARIES += lib_dyn_mpitrace.la
endif
if WANT_OPENMP
  lib_LTLIBRARIES += lib_dyn_omptrace.la
endif
endif

WRAPPERS_CORE = \
	wrappers/API/buffers.c wrappers/API/buffers.h \
	wrappers/API/wrapper.c wrappers/API/wrapper.h \
	wrappers/API/misc_wrapper.c wrappers/API/misc_wrapper.h \
	wrappers/API/trace_buffers.h \
	trace_macros.h \
	trace_hwc.h

# Wrappers for CUDA instrumentation
WRAPPERS_CUDA = wrappers/CUDA/cuda_wrapper.c wrappers/CUDA/cuda_wrapper.h \
	probes/CUDA/cuda_probe.c probes/CUDA/cuda_probe.h

# Wrappers for MPI instrumentation
WRAPPERS_MPI = mpif.h wrappers/MPI/mpi_wrapper.c wrappers/MPI/mpi_wrapper.h

# Wrappers for PACX instrumentation
WRAPPERS_PACX = wrappers/PACX/pacx_wrapper.c wrappers/PACX/pacx_wrapper.h

# Probes for OpenMP instrumentation
WRAPPERS_OMP = probes/OMP/omp_probe.c probes/OMP/omp_probe.h \
  wrappers/OMP/omp_wrapper.c wrappers/OMP/omp_wrapper.h \
  wrappers/OMP/gnu-libgomp-4.2.c wrappers/OMP/gnu-libgomp-4.2.h

if ARCH_POWERPC
if OS_LINUX
  WRAPPERS_OMP += wrappers/OMP/ibm-xlsmp-1.6.c wrappers/OMP/ibm-xlsmp-1.6.h
  EXTRA_OMP_CFLAGS = -I/usr/src/linux
endif
endif

# Wrappers for TRT instrumentation (experimental)
WRAPPERS_TRT = wrappers/TRT/trt_wrapper.c wrappers/TRT/trt_wrapper.h \
	probes/TRT/trt_probe.c probes/TRT/trt_probe.h

# Wrappers for pthreads instrumentation
WRAPPERS_PTHREAD = wrappers/pthread/pthread_wrapper.c wrappers/pthread/pthread_wrapper.h \
  probes/pthread/pthread_probe.c probes/pthread/pthread_probe.h

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
 $(COMMON_DIR)/events.h                      \
 $(COMMON_DIR)/record.h                      \
 $(COMMON_DIR)/queue.h                       \
 $(COMMON_DIR)/trace_mode.h                  \
 $(WRAPPERS_CORE)

# CUDA trace
libcudatrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_CUDA)

# MPItrace & Sequential trace
libmpitrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_MPI) $(core_MPI_SRCS)
libmpitracef_la_SOURCES = $(libmpitrace_la_SOURCES) mpif.h
libseqtrace_la_SOURCES = $(core_SRCS)

libptmpitrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_MPI) $(core_MPI_SRCS) $(WRAPPERS_PTHREAD)
libptmpitracef_la_SOURCES = $(libptmpitrace_la_SOURCES) mpif.h

# UPC (only C?)
libupctrace_la_SOURCES = $(core_SRCS)

# PACXtrace 
libpacxtrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_PACX) $(core_PACX_SRCS)
libpacxtracef_la_SOURCES = $(libpacxtrace_la_SOURCES)

# OpenMP & OpenMP + MPI
libompitrace_la_SOURCES = $(libmpitrace_la_SOURCES) $(WRAPPERS_OMP)
libompitracef_la_SOURCES = $(libmpitracef_la_SOURCES) $(WRAPPERS_OMP) mpif.h
libomptrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_OMP)

# SMPSS & SMPSS + MPI
libsmpssmpitrace_la_SOURCES = $(libmpitrace_la_SOURCES)
libsmpssmpitracef_la_SOURCES = $(libmpitrace_la_SOURCES) mpif.h
libsmpsstrace_la_SOURCES = $(core_SRCS)

# Nanos
libnanostrace_la_SOURCES = $(core_SRCS)

# pthreads & TRT
libpttrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_PTHREAD)
libtrttrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_TRT)

# LoadBalancing
libmpitrace_lb_la_SOURCES = $(libmpitrace_la_SOURCES) mpif.h
libmpitracef_lb_la_SOURCES = $(libmpitrace_la_SOURCES) mpif.h
libompitrace_lb_la_SOURCES = $(libompitrace_la_SOURCES)
libompitracef_lb_la_SOURCES = $(libompitrace_la_SOURCES) mpif.h
libsmpssmpitrace_lb_la_SOURCES = $(libmpitrace_la_SOURCES)
libsmpssmpitracef_lb_la_SOURCES = $(libmpitrace_la_SOURCES) mpif.h

# For DynInst instrumentation
lib_dyn_mpitrace_la_SOURCES = $(libmpitrace_la_SOURCES) mpif.h
lib_dyn_omptrace_la_SOURCES = $(libomptrace_la_SOURCES)

core_MODULES = \
   $(COMMON_DIR)/libcommon.la \
   $(CLOCKS_DIR)/libclock.la \
   $(HWC_DIR)/libhwc.la

INTERFACE_API_PTHREAD = $(INTERFACES_DIR)/API/libiface_api_pthread.la
if PTHREAD_SUPPORT_IN_ALL_LIBS
  INTERFACE_API = $(INTERFACE_API_PTHREAD)
else
  INTERFACE_API = $(INTERFACES_DIR)/API/libiface_api.la
endif

if HAVE_MRNET
  core_MODULES += $(MRNET_DIR)/libmrn.la
endif

if IS_BG_MACHINE
else
if WANT_MERGE_IN_TRACE
SEQUENTIAL_MERGE_LIB = $(MERGER_DIR)/libmpi2prv.la $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS)
SEQUENTIAL_CFLAGS_MERGE_LIB = -DEMBED_MERGE_IN_TRACE -I$(MERGER_DIR)/common
if HAVE_MPI
PARALLEL_MERGE_LIB = $(MERGER_DIR)/parallel/libmpimpi2prv.la $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS)
PARALLEL_MERGE_LIB_AIX = $(MERGER_DIR)/parallel/.libs/libmpimpi2prv.a $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS)
PARALLEL_CFLAGS_MERGE_LIB = -DEMBED_MERGE_IN_TRACE -I$(MERGER_DIR)/common
endif
endif
endif

libcudatrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)

libseqtrace_la_LIBADD  = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)
libmpitrace_la_LIBADD  = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpi.la $(PARALLEL_MERGE_LIB)
libmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpif.la $(PARALLEL_MERGE_LIB)
libptmpitrace_la_LIBADD  = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(INTERFACES_DIR)/MPI/libiface_mpi.la $(PARALLEL_MERGE_LIB)
libptmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(INTERFACES_DIR)/MPI/libiface_mpif.la $(PARALLEL_MERGE_LIB)

libupctrace_la_LIBADD  = $(core_MODULES) $(INTERFACE_API)

libpacxtrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/PACX/libiface_pacx.la
libpacxtracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/PACX/libiface_pacxf.la

libompitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpi.la $(PARALLEL_MERGE_LIB)
libompitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpif.la $(PARALLEL_MERGE_LIB)
libomptrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)

libsmpssmpitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpi.la $(PARALLEL_MERGE_LIB)
libsmpssmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpif.la $(PARALLEL_MERGE_LIB)
libsmpsstrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)

libnanostrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)

libpttrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API_PTHREAD) $(SEQUENTIAL_MERGE_LIB)
libtrttrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)

libmpitrace_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpi_lb.la $(PARALLEL_MERGE_LIB)
libmpitracef_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpif_lb.la $(PARALLEL_MERGE_LIB)
libompitrace_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpi_lb.la $(PARALLEL_MERGE_LIB)
libompitracef_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpif_lb.la $(PARALLEL_MERGE_LIB)

libsmpssmpitrace_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpi_lb.la $(PARALLEL_MERGE_LIB)
libsmpssmpitracef_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpif_lb.la $(PARALLEL_MERGE_LIB)

lib_dyn_mpitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/lib_dyn_iface_mpi.la $(PARALLEL_MERGE_LIB)
lib_dyn_omptrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(SEQUENTIAL_MERGE_LIB)

#
# Compilation flags
#
core_INCLUDES = \
 -I. -I$(HWC_DIR) -I$(CLOCKS_DIR) -I$(INTERFACES_DIR)/API -I$(INTERFACES_DIR)/MPI -I$(INTERFACES_DIR)/PACX \
 -I$(WRAPPERS_DIR)/API -I$(WRAPPERS_DIR)/MPI -I$(WRAPPERS_DIR)/PACX \
 -I$(WRAPPERS_DIR)/OMP -I$(WRAPPERS_DIR)/pthread -I$(WRAPPERS_DIR)/CUDA \
 -I$(top_srcdir) -I$(top_srcdir)/include -I$(COMMON_DIR) $(UNWIND_CFLAGS)
if IS_CELL_MACHINE
  core_INCLUDES += -I$(WRAPPERS_DIR)/CELL -I$(TRACER_DIR)/spu
endif
if HAVE_PAPI
  core_INCLUDES += @PAPI_CFLAGS@
endif
if HAVE_XML2
  core_INCLUDES += @XML2_CFLAGS@
endif
if HAVE_MRNET
  core_INCLUDES += -I$(MRNET_DIR)
endif
if WANT_OPENMP
  core_INCLUDES += -I$(PROBES_DIR)/OMP
endif
if WANT_PTHREAD
  core_INCLUDES += -I$(PROBES_DIR)/pthread
endif
if WANT_TRT
  core_INCLUDES += -I$(PROBES_DIR)/TRT
endif

if HAVE_PAPI
  PAPI_LINKER_FLAGS = -L@PAPI_SHAREDLIBSDIR@ -R @PAPI_SHAREDLIBSDIR@ @PAPI_LIBS@
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
if HAVE_MRNET
  MRNET_LINKER_FLAGS = -L@MRNET_LIBSDIR@ -R @MRNET_LIBSDIR@ @MRNET_LIBS@
endif

core_CFLAGS =
if PTHREAD_SUPPORT_IN_ALL_LIBS
core_CFLAGS += -DPTHREAD_SUPPORT
endif

COMMON_LINKER_FLAGS = $(HWC_LINKER_FLAGS) $(XML2_LINKER_FLAGS) $(LDL) $(UNWIND_LINKER_FLAGS) $(MRNET_LINKER_FLAGS)

libcudatrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB) -DCUDA_SUPPORT

libseqtrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) $(SEQUENTIAL_CFLAGS_MERGE_LIB)

libupctrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) -DUPC_SUPPORT

if SINGLE_MPI_LIBRARY
MPITRACE_SYMBOLS= -DC_SYMBOLS -DFORTRAN_SYMBOLS
else
MPITRACE_SYMBOLS= -DC_SYMBOLS
endif

libmpitrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ $(MPITRACE_SYMBOLS) -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB)
libmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ -DFORTRAN_SYMBOLS -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB)

libptmpitrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ $(MPITRACE_SYMBOLS) -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB) -DPTHREAD_SUPPORT
libptmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ -DFORTRAN_SYMBOLS -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB) -DPTHREAD_SUPPORT

libpacxtrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) @PACX_CFLAGS@ $(MPITRACE_SYMBOLS) -DPACX_SUPPORT
libpacxtracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @PACX_CFLAGS@ -DFORTRAN_SYMBOLS -DPACX_SUPPORT

libompitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) @MPI_CFLAGS@ $(MPITRACE_SYMBOLS) -DOMP_SUPPORT -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB)
libompitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) @MPI_CFLAGS@ -DFORTRAN_SYMBOLS -DOMP_SUPPORT -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB)
libomptrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) -DOMP_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)

libsmpssmpitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ $(MPITRACE_SYMBOLS) -DSMPSS_SUPPORT -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB)
libsmpssmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ -DFORTRAN_SYMBOLS -DSMPSS_SUPPORT -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB)
libsmpsstrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DSMPSS_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)

libnanostrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DNANOS_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)

libpttrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DPTHREAD_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)
libtrttrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DTRT_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)

libmpitrace_lb_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ $(MPITRACE_SYMBOLS) -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB)
libmpitracef_lb_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ -DFORTRAN_SYMBOLS -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB)
libompitrace_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) @MPI_CFLAGS@ $(MPITRACE_SYMBOLS) -DOMP_SUPPORT -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB)
libompitracef_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) @MPI_CFLAGS@ -DFORTRAN_SYMBOLS -DOMP_SUPPORT -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB)

libsmpssmpitrace_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ $(MPITRACE_SYMBOLS) -DSMPSS_SUPPORT -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB)
libsmpssmpitracef_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ -DFORTRAN_SYMBOLS -DSMPSS_SUPPORT -DMPI_SUPPORT $(PARALLEL_CFLAGS_MERGE_LIB)

lib_dyn_mpitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ -DC_SYMBOLS -DFORTRAN_SYMBOLS -DMPI_SUPPORT -DDYNINST_MODULE $(PARALLEL_CFLAGS_MERGE_LIB)
lib_dyn_omptrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) $(EXTRA_OMP_CFLAGS) -DDYNINST_MODULE -DOMP_SUPPORT $(SEQUENTIAL_CFLAGS_MERGE_LIB)


#
# Link flags
#
if HAVE_MPI_WITH_SHARED_LIBS
NEW_MPI_LIBS = -L@MPI_SHAREDLIBSDIR@ -R @MPI_SHAREDLIBSDIR@ @MPI_LIBS@
else
NEW_MPI_LIBS = -L@MPI_LIBSDIR@ @MPI_LIBS@ -static
endif

libcudatrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)

libseqtrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)

libupctrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)

libmpitrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(NEW_MPI_LIBS) $(COMMON_LINKER_FLAGS)
libmpitracef_la_LDFLAGS = $(libmpitrace_la_LDFLAGS) # -static 

libptmpitrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(NEW_MPI_LIBS) $(COMMON_LINKER_FLAGS) -lpthread
libptmpitracef_la_LDFLAGS = $(libptmpitrace_la_LDFLAGS) # -static 

libpacxtrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(COMMON_LINKER_FLAGS)
libpacxtracef_la_LDFLAGS = $(libpacxtrace_la_LDFLAGS) # -static

libompitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libompitracef_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libomptrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(COMMON_LINKER_FLAGS)

libsmpssmpitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libsmpssmpitracef_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libsmpsstrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(COMMON_LINKER_FLAGS)

libnanostrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(COMMON_LINKER_FLAGS)

libpttrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(COMMON_LINKER_FLAGS) -lpthread
libtrttrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(COMMON_LINKER_FLAGS)

libmpitrace_lb_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libmpitracef_lb_la_LDFLAGS = $(libmpitrace_lb_la_LDFLAGS) # -static
libompitrace_lb_la_LDFLAGS = $(libompitrace_la_LDFLAGS)
libompitracef_lb_la_LDFLAGS = $(libompitrace_la_LDFLAGS)

libsmpssmpitrace_lb_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libsmpssmpitracef_lb_la_LDFLAGS = $(libmpitrace_lb_la_LDFLAGS) # -static

lib_dyn_mpitrace_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
lib_dyn_omptrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(COMMON_LINKER_FLAGS)

## -static tells libtool to generate ONLY the static library (we don't want
##   the libmpitracef.so because it don't preloads fine with programs)
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
