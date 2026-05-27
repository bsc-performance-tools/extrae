include $(top_srcdir)/PATHS

noinst_LTLIBRARIES = libwrap_cuda.la

libwrap_cuda_la_CFLAGS = \
 -I$(COMMON_GPU_DIR) \
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
 @CUDA_CFLAGS@ @CUPTI_CFLAGS@ \
 -DCUDA_SUPPORT

libwrap_cuda_la_SOURCES = \
 cuda_wrapper_cupti.c cuda_wrapper_cupti.h

libwrap_cuda_la_LIBADD = \
  ../GPU/libwrap_gpu.la

libwrap_cuda_la_DEPENDENCIES = \
  ../GPU/libwrap_gpu.la
