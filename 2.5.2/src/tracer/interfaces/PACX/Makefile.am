include $(top_srcdir)/PATHS

iface_PACX_SRCS = pacx_interface.c pacx_interface.h
iface_PACX_INCLUDES =    \
   -I.                   \
   -I$(COMMON_DIR)       \
   -I$(TRACER_DIR)       \
   -I$(CLOCKS_DIR)       \
   -I$(HWC_DIR)          \
   -I$(WRAPPERS_DIR)/API \
   -I$(WRAPPERS_DIR)/PACX \
   -I$(INTERFACES_DIR)/API \
   @PACX_CFLAGS@ @PAPI_CFLAGS@

noinst_LTLIBRARIES = libiface_pacx.la libiface_pacxf.la

libiface_pacx_la_SOURCES = $(iface_PACX_SRCS)
libiface_pacx_la_CFLAGS = $(iface_PACX_INCLUDES) -DC_SYMBOLS -DPACX_SUPPORT

libiface_pacxf_la_SOURCES = $(iface_PACX_SRCS)
libiface_pacxf_la_CFLAGS = $(iface_PACX_INCLUDES) -DC_SYMBOLS -DFORTRAN_SYMBOLS -DPACX_SUPPORT

