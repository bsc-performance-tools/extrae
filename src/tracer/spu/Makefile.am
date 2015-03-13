include $(top_srcdir)/PATHS

CC = spu-cc
CFLAGS =

if IS_CELL_MACHINE

lib_LTLIBRARIES = libsputrace.la

libsputrace_la_SOURCES = spu_clock.c spu_clock.h \
  ../threadid.c ../threadid.h \
  ../clocks/clock.c ../clocks/clock.h \
  spu_init.c \
  spu_interface.c \
  spu_wrapper.c spu_wrapper.h \
  wrapper.c wrapper.h \
  defaults.h

libsputrace_la_CFLAGS  = -I$(TRACER_INC) -I$(SRC_DIR) -I$(COMMON_INC) -I$(CLOCKS_INC) -I$(HWC_INC)
if CELL_SDK_1
# spu-cc does not search for includes in /usr/include, but stdint.h is required
libsputrace_la_CFLAGS += -I/usr/include 
endif
libsputrace_la_LDFLAGS = -static

endif

