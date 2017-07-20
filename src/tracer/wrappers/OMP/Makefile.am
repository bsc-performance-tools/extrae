include $(top_srcdir)/PATHS

SUBDIRS = 
if WANT_OPENMP_INTEL
SUBDIRS += intel-kmpc-11-intermediate
endif

EXTRA_DIST = \
 ompt-headers
if WANT_OPENMP_GNU
EXTRA_DIST += \
 gnu-libgomp-intermediate/libgomp-doacross-intermediate-switch.c
endif

WRAPPERS_OPENMP = \
 omp-common.c omp-common.h \
 omp-probe.c omp-probe.h   \
 omp-events.c omp-events.h 

EXTRA_OPENMP_CFLAGS =

# GNU GOMP instrumentation
if WANT_OPENMP_GNU
WRAPPERS_OPENMP += gnu-libgomp.c gnu-libgomp.h
EXTRA_OPENMP_CFLAGS += -DGNU_OPENMP
endif # WANT_OPENMP_GNU

# Intel KMP instrumentation
if WANT_OPENMP_INTEL
WRAPPERS_OPENMP += intel-kmpc-11.c intel-kmpc-11.h intel-kmpc-11-intermediate/intel-kmpc-11-intermediate.h
EXTRA_OPENMP_CFLAGS += -DINTEL_OPENMP
WRAPPERS_INTEL_KMPC_11 = intel-kmpc-11-intermediate/libintel-kmpc-11-intermediate.la
endif # WANT_OPENMP_INTEL

# IBM XL SMP instrumentation
if ARCH_POWERPC
if OS_LINUX
if WANT_OPENMP_IBM
 WRAPPERS_OPENMP += ibm-xlsmp-1.6.c ibm-xlsmp-1.6.h
 EXTRA_OPENMP_CFLAGS += -I/usr/src/linux -DIBM_OPENMP
endif # ARCH_POWERPC
endif # OS_LINUX
endif # WANT_OPENMP_IBM

# OMPT support
if WANT_OPENMP_OMPT
WRAPPERS_OPENMP += \
 ompt-wrapper.c ompt-wrapper.h \
 ompt-helper.h ompt-helper.c \
 ompt-target.c ompt-target.h 
endif # WANT_OPENMP_OMPT

EXTRA_OPENMP_CFLAGS += -DOMPT_OPENMP -I$(WRAPPERS_DIR)/OMP/ompt-headers

noinst_LTLIBRARIES  = libwrap_openmp.la

libwrap_openmp_la_CFLAGS = \
 $(EXTRA_OPENMP_CFLAGS) \
 -I$(COMMON_MPI_DIR) \
 -I$(TRACER_INC) \
 -I$(HWC_INC) \
 -I$(CLOCKS_INC) \
 -I$(INTERFACES_INC)/API \
 -I$(WRAPPERS_INC)/API \
 -I$(top_srcdir) \
 -I$(top_srcdir)/include \
 -I$(COMMON_INC) \
 $(UNWIND_CFLAGS)
libwrap_openmp_la_SOURCES = $(WRAPPERS_OPENMP)
libwrap_openmp_la_LIBADD = $(WRAPPERS_INTEL_KMPC_11)
