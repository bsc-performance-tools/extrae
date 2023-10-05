include $(top_srcdir)/PATHS

# Wrappers for OPENACC instrumentation
WRAPPERS_OPENACC =

WRAPPERS_OPENACC += openacc_wrapper.h openacc_wrapper.c \
                    openacc_probe.h openacc_probe.c
WRAPPERS_OPENACC_CFLAGS = @OPENACC_CFLAGS@

noinst_LTLIBRARIES  = libwrap_openacc.la libwrap_openacccuda.la

libwrap_openacc_la_CFLAGS = \
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
 $(WRAPPERS_OPENACC_CFLAGS)

libwrap_openacc_la_LDFLAGS = $(WRAPPERS_OPENACC_LDFLAGS)

libwrap_openacc_la_SOURCES = $(WRAPPERS_OPENACC)

WRAPPERS_OPENACCCUDA_CFLAGS =

if HAVE_CUPTI 
WRAPPERS_OPENACCCUDA_CFLAGS += @CUPTI_CFLAGS@
endif

if HAVE_CUDA
libwrap_openacccuda_la_SOURCES = $(WRAPPERS_OPENACC)
libwrap_openacccuda_la_CFLAGS = $(libwrap_openacc_la_CFLAGS) @CUDA_CFLAGS@ $(WRAPPERS_OPENACCCUDA_CFLAGS) -I$(WRAPPERS_INC)/CUDA -DCUDA_SUPPORT
libwrap_openacccuda_la_LDFLAGS = $(libwrap_openacc_la_LDFLAGS)
endif

