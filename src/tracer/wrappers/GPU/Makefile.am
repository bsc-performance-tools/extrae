include $(top_srcdir)/PATHS

# Common GPU wrapper code shared by CUDA/HIP
WRAPPERS_GPU = \
 gpu_macros.h \
 gpu_probe.h gpu_probe.c \
 gpu_common.h gpu_common.c

if USE_LEGACY_GPU_WRAPPER
WRAPPERS_GPU += gpu_wrapper.c gpu_wrapper.h
endif

noinst_LTLIBRARIES = libwrap_gpu.la

libwrap_gpu_la_CFLAGS = \
 -I$(COMMON_MPI_DIR) \
 -I$(TRACER_INC) \
 -I$(HWC_INC) \
 -I$(CLOCKS_INC) \
 -I$(INTERFACES_INC)/API \
 -I$(WRAPPERS_INC)/API \
 -I$(WRAPPERS_INC)/CUDA \
 -I$(WRAPPERS_INC)/HIP \
 -I$(top_srcdir) \
 -I$(top_srcdir)/include \
 -I$(COMMON_INC) \
 $(UNWIND_CFLAGS)

if HAVE_HIP
libwrap_gpu_la_CFLAGS += $(HIP_CFLAGS) -DHIP_SUPPORT
endif

if HAVE_CUDA
libwrap_gpu_la_CFLAGS += $(CUDA_CFLAGS) -DCUDA_SUPPORT
endif

libwrap_gpu_la_SOURCES = $(WRAPPERS_GPU)