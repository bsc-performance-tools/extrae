include $(top_srcdir)/PATHS

SUBDIRS = clocks hwc interfaces

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

if HAVE_PACX
  lib_LTLIBRARIES += libpacxtrace.la libpacxtracef.la
endif

if HAVE_MPI
  lib_LTLIBRARIES += libmpitrace.la libmpitracef.la
if WANT_OPENMP
  lib_LTLIBRARIES += libompitrace.la
endif
if WANT_SMPSS
  lib_LTLIBRARIES += libsmpssmpitrace.la libsmpssmpitracef.la
endif
endif

if WANT_OPENMP
  lib_LTLIBRARIES += libomptrace.la
endif
if WANT_SMPSS
  lib_LTLIBRARIES += libsmpsstrace.la
endif
if WANT_PTHREAD
  lib_LTLIBRARIES += libpttrace.la
endif

if GENERATE_LOAD_BALANCING
if HAVE_MPI
  lib_LTLIBRARIES += libmpitrace-lb.la libmpitracef-lb.la
if WANT_OPENMP
  lib_LTLIBRARIES += libompitrace-lb.la
endif
if WANT_SMPSS
  lib_LTLIBRARIES += libsmpssmpitrace-lb.la libsmpssmpitracef-lb.la
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
	wrappers/API/buffers.c \
	wrappers/API/wrapper.c \
	wrappers/API/misc_wrapper.c

# Wrappers for MPI instrumentation
WRAPPERS_MPI = wrappers/MPI/mpi_wrapper.c

# Wrappers for PACX instrumentation
WRAPPERS_PACX = wrappers/PACX/pacx_wrapper.c

# Probes for OpenMP instrumentation
WRAPPERS_OMP = probes/OMP/omp_probe.c \
  wrappers/OMP/omp_wrapper.c \
  wrappers/OMP/gnu-libgomp-4.2.c

if ARCH_POWERPC
if OS_LINUX
  WRAPPERS_OMP += wrappers/OMP/ibm-xlsmp-1.6.c
endif
endif

# Wrappers for TRT instrumentation (experimental)
WRAPPERS_TRT = wrappers/TRT/trt_wrapper.c \
	probes/TRT/trt_probe.c

# Wrappers for pthreads instrumentation
WRAPPERS_PTHREAD = wrappers/pthread/pthread_wrapper.c \
  probes/pthread/pthread_probe.c

if IS_CELL_MACHINE
  WRAPPERS_CORE += wrappers/CELL/cell_wrapper.c
endif

core_MPI_SRCS = \
 hash_table.c hash_table.h                   \
 persistent_requests.c persistent_requests.h

core_PACX_SRCS = $(core_MPI_SRCS)

core_SRCS = \
 calltrace.c calltrace.h                     \
 signals.c signals.h                         \
 xml-parse.c xml-parse.h                     \
 UF_gcc_instrument.c UFF_gcc_instruments.h   \
 mode.c mode.h                               \
 threadid.h threadid.c                       \
 taskid.h taskid.c                           \
 uint64.h defines.h                          \
 $(COMMON_DIR)/events.h                      \
 $(COMMON_DIR)/record.h                      \
 $(COMMON_DIR)/trace_macros.h                \
 $(COMMON_DIR)/queue.h                       \
 $(COMMON_DIR)/trace_mode.h                  \
 $(WRAPPERS_CORE)

# MPItrace & Sequential trace
libmpitrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_MPI) $(core_MPI_SRCS) mpif.h
libmpitracef_la_SOURCES = $(libmpitrace_la_SOURCES) mpif.h
libseqtrace_la_SOURCES = $(core_SRCS)

# PACXtrace 
libpacxtrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_PACX) $(core_PACX_SRCS)
libpacxtracef_la_SOURCES = $(libpacxtrace_la_SOURCES)

# OpenMP & OpenMP + MPI
libompitrace_la_SOURCES = $(libmpitrace_la_SOURCES) $(WRAPPERS_OMP)
libomptrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_OMP)

# SMPSS & SMPSS + MPI
libsmpssmpitrace_la_SOURCES = $(libmpitrace_la_SOURCES)
libsmpssmpitracef_la_SOURCES = $(libmpitrace_la_SOURCES) mpif.h
libsmpsstrace_la_SOURCES = $(core_SRCS)

# pthreads & TRT
libpttrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_PTHREAD)
libtrttrace_la_SOURCES = $(core_SRCS) $(WRAPPERS_TRT)

# LoadBalancing
libmpitrace_lb_la_SOURCES = $(libmpitrace_la_SOURCES) mpif.h
libmpitracef_lb_la_SOURCES = $(libmpitrace_la_SOURCES) mpif.h
libompitrace_lb_la_SOURCES = $(libompitrace_la_SOURCES)
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

libseqtrace_la_LIBADD  = $(core_MODULES) $(INTERFACE_API)
libmpitrace_la_LIBADD  = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpi.la
libmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpif.la

libpacxtrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/PACX/libiface_pacx.la
libpacxtracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/PACX/libiface_pacxf.la

libompitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpi.la
libomptrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API)

libsmpssmpitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpi.la
libsmpssmpitracef_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpif.la
libsmpsstrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API)

libpttrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API_PTHREAD)
libtrttrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API)

libmpitrace_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpi_lb.la
libmpitracef_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpif_lb.la
libompitrace_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpi_lb.la

libsmpssmpitrace_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpi_lb.la
libsmpssmpitracef_lb_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/libiface_mpif_lb.la

lib_dyn_mpitrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API) $(INTERFACES_DIR)/MPI/lib_dyn_iface_mpi.la
lib_dyn_omptrace_la_LIBADD = $(core_MODULES) $(INTERFACE_API)

#
# Compilation flags
#
core_INCLUDES = \
 -I. -I$(HWC_DIR) -I$(CLOCKS_DIR) -I$(INTERFACES_DIR)/API -I$(INTERFACES_DIR)/MPI -I$(INTERFACES_DIR)/PACX \
 -I$(WRAPPERS_DIR)/API -I$(WRAPPERS_DIR)/MPI -I$(WRAPPERS_DIR)/PACX \
 -I$(WRAPPERS_DIR)/OMP -I$(WRAPPERS_DIR)/pthread \
 -I$(top_srcdir)/include -I$(COMMON_DIR)
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
else
  PAPI_LINKER_FLAGS =
endif
if HAVE_PMAPI
  PMAPI_LINKER_FLAGS = -lpmapi
else
  PMAPI_LINKER_FLAGS =
endif
HWC_LINKER_FLAGS = $(PAPI_LINKER_FLAGS) $(PMAPI_LINKER_FLAGS)

if HAVE_XML2
  XML2_LINKER_FLAGS = -L@XML2_SHAREDLIBSDIR@ -R @XML2_SHAREDLIBSDIR@ @XML2_LIBS@
else
  XML2_LINKER_FLAGS =
endif
if HAVE_MRNET
  MRNET_LINKER_FLAGS = -L@MRNET_LIBSDIR@ -R @MRNET_LIBSDIR@ @MRNET_LIBS@
else
  MRNET_LINKER_FLAGS = 
endif
if HAVE_LIBDL
  LDL = -ldl
else
  LDL =
endif

core_CFLAGS =
if PTHREAD_SUPPORT_IN_ALL_LIBS
core_CFLAGS += -DPTHREAD_SUPPORT
endif

COMMON_LINKER_FLAGS = $(HWC_LINKER_FLAGS) $(XML2_LINKER_FLAGS) $(LDL) $(MRNET_LINKER_FLAGS)

libseqtrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS)

if SINGLE_MPI_LIBRARY
MPITRACE_SYMBOLS= -DC_SYMBOLS -DFORTRAN_SYMBOLS
else
MPITRACE_SYMBOLS= -DC_SYMBOLS
endif
libmpitrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ $(MPITRACE_SYMBOLS) -DMPI_SUPPORT
libmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ -DFORTRAN_SYMBOLS -DMPI_SUPPORT

libpacxtrace_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) @PACX_CFLAGS@ $(MPITRACE_SYMBOLS) -DPACX_SUPPORT
libpacxtracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @PACX_CFLAGS@ -DFORTRAN_SYMBOLS -DPACX_SUPPORT

libompitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ $(MPITRACE_SYMBOLS) -DOMP_SUPPORT -DMPI_SUPPORT
libomptrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DOMP_SUPPORT

libsmpssmpitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ $(MPITRACE_SYMBOLS) -DSMPSS_SUPPORT -DMPI_SUPPORT
libsmpssmpitracef_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ -DFORTRAN_SYMBOLS -DSMPSS_SUPPORT -DMPI_SUPPORT
libsmpsstrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DSMPSS_SUPPORT

libpttrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DPTHREAD_SUPPORT
libtrttrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DTRT_SUPPORT

libmpitrace_lb_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ $(MPITRACE_SYMBOLS) -DMPI_SUPPORT
libmpitracef_lb_la_CFLAGS  = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ -DFORTRAN_SYMBOLS -DMPI_SUPPORT
libompitrace_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ $(MPITRACE_SYMBOLS) -DOMP_SUPPORT -DMPI_SUPPORT

libsmpssmpitrace_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ $(MPITRACE_SYMBOLS) -DSMPSS_SUPPORT -DMPI_SUPPORT
libsmpssmpitracef_lb_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ -DFORTRAN_SYMBOLS -DSMPSS_SUPPORT -DMPI_SUPPORT

lib_dyn_mpitrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) @MPI_CFLAGS@ -DC_SYMBOLS -DMPI_SUPPORT -DDYNINST_MODULE
lib_dyn_omptrace_la_CFLAGS = $(core_INCLUDES) $(core_CFLAGS) -DDYNINST_MODULE -DOMP_SUPPORT

#
# Link flags
#
if HAVE_MPI_WITH_SHARED_LIBS
NEW_MPI_LIBS = -L@MPI_SHAREDLIBSDIR@ -R @MPI_SHAREDLIBSDIR@ @MPI_LIBS@
else
NEW_MPI_LIBS = -L@MPI_LIBSDIR@ @MPI_LIBS@ -static
endif

libseqtrace_la_LDFLAGS = $(COMMON_LINKER_FLAGS)

libmpitrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(NEW_MPI_LIBS) $(COMMON_LINKER_FLAGS)
libmpitracef_la_LDFLAGS = $(libmpitrace_la_LDFLAGS) -static

libpacxtrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(COMMON_LINKER_FLAGS)
libpacxtracef_la_LDFLAGS = $(libpacxtrace_la_LDFLAGS) -static

libompitrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(NEW_MPI_LIBS) $(COMMON_LINKER_FLAGS)
libomptrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(COMMON_LINKER_FLAGS)

libsmpssmpitrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(NEW_MPI_LIBS) $(COMMON_LINKER_FLAGS)
libsmpssmpitracef_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(NEW_MPI_LIBS) $(COMMON_LINKER_FLAGS) -static
libsmpsstrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(COMMON_LINKER_FLAGS)

libpttrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(COMMON_LINKER_FLAGS) -lpthread
libtrttrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(COMMON_LINKER_FLAGS)

libmpitrace_lb_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libmpitracef_lb_la_LDFLAGS = $(libmpitrace_lb_la_LDFLAGS) -static
libompitrace_lb_la_LDFLAGS = $(libompitrace_la_LDFLAGS)

libsmpssmpitrace_lb_la_LDFLAGS = $(libmpitrace_la_LDFLAGS)
libsmpssmpitracef_lb_la_LDFLAGS = $(libmpitrace_lb_la_LDFLAGS) -static

lib_dyn_mpitrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(COMMON_LINKER_FLAGS)
lib_dyn_omptrace_la_LDFLAGS = -no-undefined -version-info 1:0:0 $(COMMON_LINKER_FLAGS)

## -static tells libtool to generate ONLY the static library (we don't want
##   the libmpitracef.so because it don't preloads fine with programs)
## -R flag works like -Wl,-rpath
## -no-undefined could be interesting for AIX?
##   check http://www.andamooka.org/reader.pl?pgid=autobookautobook_89

install-exec-hook:
if IS_CELL_MACHINE
	cd $(libdir); $(LN_S) -f libseqtrace.a libpputrace.a
endif

## These are the options of the previous Makefile (from Imakefile) which are
## not still ported to the configure system!

#### -DDEBUG_COMMUNICATORS mostra informacio sobre els comunicadors.
####
#### -DDEBUG_MPITRACE Prints debug information of the "internals"
