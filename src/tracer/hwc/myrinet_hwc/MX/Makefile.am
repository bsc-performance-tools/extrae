
lib_LTLIBRARIES = libmx_counters.la

libmx_counters_la_SOURCES = \
	mx_counters.c   \
	mx_counters.h

libmx_counters_la_CFLAGS  = \
	@MX_CFLAGS@                               \
	-I@MX_INCLUDES@                           \
	-I@MX_INCLUDES@/mx_internals              \
	-I../include/mpich-mx-1.2.6..0.94/include \
	-I../include/mpich-mx-1.2.6..0.94/mpid/ch_mx

libmx_counters_la_LDFLAGS = -L@MX_LIBSDIR@ -lmyriexpress

