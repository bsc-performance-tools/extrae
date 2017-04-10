include $(top_srcdir)/PATHS

# Wrappers for CUDA instrumentation
WRAPPERS_CUDA = \
 cuda_probe.c cuda_probe.h \
 cuda_common.c cuda_common.h

if WANT_CUDA
WRAPPERS_CUDA += cuda_wrapper.c cuda_wrapper.h
WRAPPERS_CUDA_CFLAGS = @CUDA_CFLAGS@
endif
if WANT_CUPTI
WRAPPERS_CUDA += cuda_wrapper_cupti.c cuda_wrapper_cupti.h
WRAPPERS_CUDA_CFLAGS = @CUPTI_CFLAGS@ @CUDA_CFLAGS@
WRAPPERS_CUDA_LDFLAGS = @CUPTI_LDFLAGS@ -lcupti
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

libwrap_cuda_la_LDFLAGS = $(WRAPPERS_CUDA_LDFLAGS)

libwrap_cuda_la_SOURCES = $(WRAPPERS_CUDA)

