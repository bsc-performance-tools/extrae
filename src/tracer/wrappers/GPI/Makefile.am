include $(top_srcdir)/PATHS

WRAPPERS_GPI = \
    gpi_wrapper.h gpi_wrapper.c \
	gpi_events.h gpi_events.c \
	gpi_probe.h gpi_probe.c

EXTRA_GPI_CFLAGS = \
	-DGPI2_WITH_MPI \
	-I$(GPI_INCLUDES) \
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

noinst_LTLIBRARIES = libwrap_gpi.la

libwrap_gpi_la_CFLAGS = -DC_SYMBOLS $(EXTRA_GPI_CFLAGS)
libwrap_gpi_la_SOURCES = $(WRAPPERS_GPI)
