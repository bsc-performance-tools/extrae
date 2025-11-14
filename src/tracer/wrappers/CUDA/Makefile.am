include $(top_srcdir)/PATHS

# Wrappers for CUDA instrumentation
WRAPPERS_CUDA = \
 cuda_probe.c cuda_probe.h \
 cuda_common.c cuda_common.h \
 gpu_event_info.c gpu_event_info.h

if HAVE_CUDA 
WRAPPERS_CUDA_CFLAGS = @CUDA_CFLAGS@
WRAPPERS_CUDA_LIBS   = -lcuda
if HAVE_CUPTI
WRAPPERS_CUDA += cuda_wrapper_cupti.c cuda_wrapper_cupti.h
WRAPPERS_CUDA_CFLAGS += @CUPTI_CFLAGS@ 
else 
WRAPPERS_CUDA += cuda_wrapper.c cuda_wrapper.h
endif
endif

noinst_LTLIBRARIES  = libwrap_cuda.la

libwrap_cuda_la_CFLAGS = \
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
 $(WRAPPERS_CUDA_CFLAGS)

libwrap_cuda_la_SOURCES = $(WRAPPERS_CUDA)

libwrap_cuda_la_LIBADD = $(WRAPPERS_CUDA_LIBS)