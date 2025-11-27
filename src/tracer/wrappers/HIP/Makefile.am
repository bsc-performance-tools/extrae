include $(top_srcdir)/PATHS

# Wrappers for HIP instrumentation
WRAPPERS_HIP = \
 hip_probe.c hip_probe.h \
 hip_common.c hip_common.h \
 gpu_event_info.c gpu_event_info.h

WRAPPERS_HIP_CFLAGS = @HIP_CFLAGS@
WRAPPERS_HIP += hip_wrapper_roctracer.c hip_wrapper_roctracer.h


noinst_LTLIBRARIES  = libwrap_hip.la

libwrap_hip_la_CFLAGS = \
 -I$(COMMON_MPI_DIR) \
 -I$(TRACER_INC) \
 -I$(HWC_INC) \
 -I$(CLOCKS_INC) \
 -I$(INTERFACES_INC)/API \
 -I$(WRAPPERS_INC)/API \
 -I$(top_srcdir) \
 -I$(top_srcdir)/include \
 -I$(COMMON_INC) \
 $(UNWIND_CFLAGS) \
 $(WRAPPERS_HIP_CFLAGS)

libwrap_hip_la_CFLAGS += -D__HIP_PLATFORM_AMD__
libwrap_hip_la_SOURCES = $(WRAPPERS_HIP)

