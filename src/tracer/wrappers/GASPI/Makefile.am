include $(top_srcdir)/PATHS

WRAPPERS_GASPI = \
    gaspi_wrapper.h gaspi_wrapper.c \
	gaspi_events.h gaspi_events.c \
	gaspi_probe.h gaspi_probe.c

EXTRA_GASPI_CFLAGS = \
	-I$(GASPI_INCLUDES) \
	-I$(COMMON_MPI_DIR) \
	-I$(TRACER_INC) \
	-I$(HWC_INC) \
	-I$(CLOCKS_INC) \
	-I$(INTERFACES_INC)/API \
	-I$(WRAPPERS_INC)/API \
	-I$(top_srcdir) \
	-I$(top_srcdir)/include \
	-I$(COMMON_INC) \
	$(UNWIND_CFLAGS)

noinst_LTLIBRARIES = libwrap_gaspi.la

libwrap_gaspi_la_CFLAGS = -DC_SYMBOLS $(EXTRA_GASPI_CFLAGS)
libwrap_gaspi_la_SOURCES = $(WRAPPERS_GASPI)
