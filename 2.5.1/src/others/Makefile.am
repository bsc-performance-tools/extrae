bin_PROGRAMS = genmpits reducempit

if HAVE_PAPI
bin_PROGRAMS += papi_best_set
endif

genmpits_SOURCES = genmpits.c
genmpits_CFLAGS = -I. -I../common

reducempit_SOURCES = reducempit.c
reducempit_CFLAGS = -I../.. -I../common

papi_best_set_SOURCES = papi_best_set.c
papi_best_set_CFLAGS  = -I. -I../common -DPAPI_HOME='"$(PAPI_HOME)"' @PAPI_CFLAGS@
papi_best_set_LDFLAGS = @PAPI_LDFLAGS@ @PAPI_LIBS@
if IS_BG_MACHINE
papi_best_set_LDFLAGS += -all-static
else
papi_best_set_LDFLAGS += -L@PAPI_SHAREDLIBSDIR@ -R @PAPI_SHAREDLIBSDIR@ @PAPI_LIBS@
endif
