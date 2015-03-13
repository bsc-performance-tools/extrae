include $(top_srcdir)/PATHS

iface_PACX_SRCS = pacx_interface.c pacx_interface.h
iface_PACX_INCLUDES =    \
   -I$(INTERFACES_INC)/PACX \
   -I$(COMMON_INC)       \
   -I$(TRACER_INC)       \
   -I$(CLOCKS_INC)       \
   -I$(HWC_INC)          \
   -I$(WRAPPERS_INC)/API \
   -I$(WRAPPERS_INC)/PACX \
   -I$(INTERFACES_INC)/API \
   @PACX_CFLAGS@ @PAPI_CFLAGS@

noinst_LTLIBRARIES = libiface_pacx.la libiface_pacxf.la

libiface_pacx_la_SOURCES = $(iface_PACX_SRCS)
libiface_pacx_la_CFLAGS = $(iface_PACX_INCLUDES) -DC_SYMBOLS -DPACX_SUPPORT

libiface_pacxf_la_SOURCES = $(iface_PACX_SRCS)
libiface_pacxf_la_CFLAGS = $(iface_PACX_INCLUDES) -DC_SYMBOLS -DFORTRAN_SYMBOLS -DPACX_SUPPORT

