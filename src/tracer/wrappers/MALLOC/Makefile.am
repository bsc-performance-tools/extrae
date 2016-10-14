include $(top_srcdir)/PATHS

# Wrappers for malloc instrumentation
WRAPPERS_MALLOC = \
 malloc_wrapper.c malloc_wrapper.h \
 malloc_probe.c malloc_probe.h 

noinst_LTLIBRARIES  = libwrap_malloc.la

libwrap_malloc_la_CFLAGS = \
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

if HAVE_MEMKIND
libwrap_malloc_la_CFLAGS += ${MEMKIND_CFLAGS}
libwrap_malloc_la_LDFLAGS = ${MEMKIND_LDFLAGS} ${MEMKIND_LIBS}
endif

libwrap_malloc_la_SOURCES = $(WRAPPERS_MALLOC)

