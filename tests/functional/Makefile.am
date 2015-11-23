SUBDIRS = \
 launcher \
 tracer \
 merger \
 callstack

if HAVE_PAPI
SUBDIRS += hw-counters
endif

if HAVE_XML2
SUBDIRS += xml
endif

EXTRA_DIST = helper_functions.bash
