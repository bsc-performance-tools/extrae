include $(top_srcdir)/PATHS

EXTRA_DIST = \
 intel-kmpc-11-intermediate-switch.c \
 intel-kmpc-11-intermediate.c \
 ompt-headers

WRAPPERS_OPENMP = \
 omp_probe.c omp_probe.h \
 omp-common.c omp-common.h 

EXTRA_OPENMP_CFLAGS =

if WANT_OPENMP_OMPT
WRAPPERS_OPENMP += \
 ompt-wrapper.c \
 ompt-helper.h ompt-helper.c \
 ompt-target.c ompt-target.h 

EXTRA_OPENMP_CFLAGS += -DOMPT_OPENMP -I$(WRAPPERS_DIR)/OMP/ompt-headers

else

# These files are needed by intel/ibm/gnu openmp instrumentation, but not by OMPT
WRAPPERS_OPENMP += omp_wrapper.c omp_wrapper.h

if WANT_OPENMP_INTEL
WRAPPERS_OPENMP += intel-kmpc-11.c intel-kmpc-11.h intel-kmpc-11-intermediate.h
EXTRA_OPENMP_CFLAGS += -DINTEL_OPENMP
endif # WANT_OPENMP_INTEL

if WANT_OPENMP_GNU
if WANT_OPENMP_GNU_4_2
WRAPPERS_OPENMP += gnu-libgomp-4.2.c gnu-libgomp-4.2.h
EXTRA_OPENMP_CFLAGS += -DGNU_OPENMP -DGNU_OPENMP_4_2
endif # WANT_OPENMP_GNU_4_2
if WANT_OPENMP_GNU_4_9
WRAPPERS_OPENMP += gnu-libgomp-4.9.c gnu-libgomp-4.9.h
EXTRA_OPENMP_CFLAGS += -DGNU_OPENMP -DGNU_OPENMP_4_9
endif # WANT_OPENMP_GNU_4_9
endif # WANT_OPENMP_GNU

if ARCH_POWERPC
if OS_LINUX
if WANT_OPENMP_IBM
 WRAPPERS_OPENMP += ibm-xlsmp-1.6.c ibm-xlsmp-1.6.h
 EXTRA_OPENMP_CFLAGS += -I/usr/src/linux -DIBM_OPENMP
endif # ARCH_POWERPC
endif # OS_LINUX
endif # WANT_OPENMP_IBM

endif # WANT_OPENMP_OMPT


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

