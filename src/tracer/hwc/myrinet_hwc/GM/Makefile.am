
lib_LTLIBRARIES = libgm_counters.la

libgm_counters_la_SOURCES = \
	gm_counters.c   \
	gm_counters.h

libgm_counters_la_CFLAGS  = \
	-I@GM_INCLUDES@                       \
	-I../include/gm-2.0.21_MacOSX/include \
	-I../include/gm-2.0.21_MacOSX/drivers/linux/gm
	
libgm_counters_la_LDFLAGS = -L@GM_LIBSDIR@ -lgm

