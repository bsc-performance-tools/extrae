bin_PROGRAMS = genmpits reducempit

if HAVE_PAPI
bin_PROGRAMS += papi_best_set_old papi_best_set
endif

genmpits_SOURCES = genmpits.c
genmpits_CFLAGS = -I. -I../common

reducempit_SOURCES = reducempit.c
reducempit_CFLAGS = -I../.. -I../common

papi_best_set_old_SOURCES = papi_best_set_old.c
papi_best_set_old_CFLAGS  = -I. -I../common @PAPI_CFLAGS@
papi_best_set_SOURCES = papi_best_set.C
papi_best_set_CXXFLAGS  = -I. -I../common @PAPI_CFLAGS@
if !HAVE_PAPI_EMBEDDED
  papi_best_set_old_CFLAGS += -DPAPI_HOME='"$(PAPI_HOME)"'
  papi_best_set_CXXFLAGS += -DPAPI_HOME='"$(PAPI_HOME)"'
else
  papi_best_set_old_CFLAGS += -DPAPI_HOME='"embedded"'
  papi_best_set_CXXFLAGS += -DPAPI_HOME='"embedded"'
endif
papi_best_set_old_LDFLAGS = @PAPI_LDFLAGS@ @PAPI_LIBS@
papi_best_set_LDFLAGS = @PAPI_LDFLAGS@ @PAPI_LIBS@
if IS_BG_MACHINE
papi_best_set_old_LDFLAGS += -all-static
papi_best_set_LDFLAGS += -all-static
else
if !HAVE_PAPI_EMBEDDED
papi_best_set_old_LDFLAGS += -L@PAPI_SHAREDLIBSDIR@ -R @PAPI_SHAREDLIBSDIR@ @PAPI_LIBS@
papi_best_set_LDFLAGS += -L@PAPI_SHAREDLIBSDIR@ -R @PAPI_SHAREDLIBSDIR@ @PAPI_LIBS@
endif
endif

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
	$(MKDIR_P) $(prefix)/lib
	cp $(PYTHON_MODULE) $(prefix)/lib

clean-local:
	rm -f $(PYTHON_MODULE)
