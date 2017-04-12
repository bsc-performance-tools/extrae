include $(top_srcdir)/PATHS

CLOCKS = clock.c clock.h posix_clock.c posix_clock.h rusage_clock.c rusage_clock.h
if USE_GETTIMEOFDAY_CLOCK
  CLOCKS += gettimeofday_clock.c gettimeofday_clock.h
else
if !USE_POSIX_CLOCK
if IS_BG_MACHINE
if IS_BGL_MACHINE
  CLOCKS += bgl_clock.c bgl_clock.h
endif
if IS_BGP_MACHINE
  CLOCKS += bgp_clock.c bgp_clock.h
endif
if IS_BGQ_MACHINE
  CLOCKS += bgq_clock.c bgq_clock.h
endif
else
if ARCH_IA32
  CLOCKS += ia32_clock.c ia32_clock.h
endif
if ARCH_IA64
  CLOCKS += ia64_clock.c ia64_clock.h
endif
if ARCH_POWERPC
  CLOCKS += ppc_clock.c ppc_clock.h
endif
endif
endif
endif

noinst_LTLIBRARIES = libclock.la
libclock_la_SOURCES = $(CLOCKS)
libclock_la_CFLAGS  = -I$(CLOCKS_INC) -I$(COMMON_INC) -I$(TRACER_INC)
libclock_la_LDFLAGS = -lrt
