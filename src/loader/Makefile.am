include $(top_srcdir)/PATHS

bin_PROGRAMS = extrae-loader

extrae_loader_SOURCES = \
  extrae-loader.c

if !IS_GR740_MACHINE
extrae_loader_LDFLAGS = -ldl
endif

