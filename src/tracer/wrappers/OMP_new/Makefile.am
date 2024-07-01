include $(top_srcdir)/PATHS

EXTRA_DIST = 

if WANT_OPENMP_GNU
EXTRA_DIST += \
 gnu-libgomp-intermediate/libgomp-doacross-intermediate-switch.c
endif

WRAPPERS_OPENMP = \
 omp_common.c omp_common.h \
 omp_events.c omp_events.h \
 nested_id.c nested_id.h \
 atomic_counter.h \
 wrap_macros.h \
 wrap_checkpoints.h \
 omp_wrap_macros.h \
 pdebug.h

EXTRA_OPENMP_CFLAGS =

# GNU GOMP instrumentation
if WANT_OPENMP_GNU
WRAPPERS_OPENMP += gnu_libgomp.c gnu_libgomp.h gomp_helpers.c gomp_helpers.h gomp_wrappers.c gomp_probes.c gomp_probes.h gomp_probes_burst.c gomp_probes_burst.h
EXTRA_OPENMP_CFLAGS += -DGNU_OPENMP
endif # WANT_OPENMP_GNU

noinst_LTLIBRARIES  = libwrap_openmp.la

libwrap_openmp_la_CFLAGS = \
 $(EXTRA_OPENMP_CFLAGS) \
 -I$(TRACER_INC) \
 -I$(HWC_INC) \
 -I$(CLOCKS_INC) \
 -I$(WRAPPERS_INC)/API \
 -I$(top_srcdir) \
 -I$(top_srcdir)/include \
 -I$(COMMON_INC) \
 -I$(STATS_INC) \
 -I$(STATS_INC)/OMP \
 -I$(BURST_INC)
libwrap_openmp_la_SOURCES = $(WRAPPERS_OPENMP)
