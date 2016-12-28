include $(top_srcdir)/PATHS

WRAPPERS_SYSCALL = \
 syscall_wrapper.c syscall_wrapper.h \
 syscall_probe.c   syscall_probe.h 

noinst_LTLIBRARIES  = libwrap_syscall.la

libwrap_syscall_la_CFLAGS =   \
 -I$(COMMON_MPI_DIR)     \
 -I$(TRACER_INC)         \
 -I$(HWC_INC)            \
 -I$(CLOCKS_INC)         \
 -I$(INTERFACES_INC)/API \
 -I$(WRAPPERS_INC)/API   \
 -I$(top_srcdir)         \
 -I$(top_srcdir)/include \
 -I$(COMMON_INC)         \
 $(UNWIND_CFLAGS)

libwrap_syscall_la_SOURCES = $(WRAPPERS_SYSCALL)

