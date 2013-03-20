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
papi_best_set_LDFLAGS = -all-static @PAPI_LDFLAGS@ @PAPI_LIBS@

PYTHON_MODULE = pyextrae.py
noinst_SCRIPTS = $(PYTHON_MODULE)
EXTRA_DIST = $(PYTHON_MODULE).in

$(PYTHON_MODULE): $(PYTHON_MODULE).in
	cp $^ $@
	$(top_srcdir)/substitute $(SED) "@sub_PREFIX@" ${prefix} $@
if HAVE_PAPI
	$(top_srcdir)/substitute $(SED) "@sub_Extrae_eventandcounters@" Extrae_eventandcounters $@
else
	$(top_srcdir)/substitute $(SED) "@sub_Extrae_eventandcounters@" Extrae_event $@
endif

install-data-hook:
	$(mkdir_p) $(prefix)/lib
	cp $(PYTHON_MODULE) $(prefix)/lib

clean-local:
	rm -f $(PYTHON_MODULE)
