include $(top_srcdir)/PATHS

noinst_LTLIBRARIES = libwrap_hip.la

libwrap_hip_la_CFLAGS = \
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
 @HIP_CFLAGS@ @ROCTRACER_CFLAGS@ \
 -D__HIP_PLATFORM_AMD__ -DHIP_SUPPORT

libwrap_hip_la_SOURCES = \
 hip_wrapper_roctracer.c hip_wrapper_roctracer.h

libwrap_hip_la_LIBADD = \
  ../GPU/libwrap_gpu.la \
  @ROCTRACER_LDFLAGS_RPATH@

libwrap_hip_la_DEPENDENCIES = \
  ../GPU/libwrap_gpu.la
