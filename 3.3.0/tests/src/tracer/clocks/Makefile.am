include $(top_srcdir)/PATHS

check_PROGRAMS = 
TESTS = 

if HAVE_SLEEP
 check_PROGRAMS += check_clock_sleep
 TESTS += check_clock_sleep
endif

if HAVE_USLEEP
 check_PROGRAMS += check_clock_usleep
 TESTS += check_clock_usleep
endif

if HAVE_NANOSLEEP
 check_PROGRAMS += check_clock_nanosleep
 TESTS += check_clock_nanosleep
endif


# POSIX clock may be available by default. Avoid adding these if unneeded.
if USE_POSIX_CLOCK
if NEED_POSIX_CLOCK 
  EXTRA = -L@RT_LIBSDIR@ -R @RT_LIBSDIR@ @RT_LIBS@
endif
endif

if HAVE_SLEEP
 check_clock_sleep_SOURCES = check_clock_sleep.c
 check_clock_sleep_CFLAGS = -I$(CLOCKS_INC) -I$(COMMON_INC) -I$(TRACER_INC)
 check_clock_sleep_LDFLAGS = $(CLOCKS_LIB)/libclock.la $(EXTRA)
endif

if HAVE_USLEEP
 check_clock_usleep_SOURCES = check_clock_usleep.c
 check_clock_usleep_CFLAGS = -I$(CLOCKS_INC) -I$(COMMON_INC) -I$(TRACER_INC)
 check_clock_usleep_LDFLAGS = $(CLOCKS_LIB)/libclock.la $(EXTRA)
endif

if HAVE_NANOSLEEP
 check_clock_nanosleep_SOURCES = check_clock_nanosleep.c
 check_clock_nanosleep_CFLAGS = -I$(CLOCKS_INC) -I$(COMMON_INC) -I$(TRACER_INC)
 check_clock_nanosleep_LDFLAGS = $(CLOCKS_LIB)/libclock.la $(EXTRA)
endif
