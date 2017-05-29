bin_PROGRAMS = genmpits reducempit

if HAVE_PAPI
bin_PROGRAMS += papi_best_set_old papi_best_set
endif

genmpits_SOURCES = genmpits.c
genmpits_CFLAGS = -I$(COMMON_INC)

reducempit_SOURCES = reducempit.c
reducempit_CFLAGS = -I$(COMMON_INC)

papi_best_set_old_SOURCES = papi_best_set_old.c
papi_best_set_old_CFLAGS  = -I$(COMMON_INC) @PAPI_CFLAGS@
papi_best_set_SOURCES = papi_best_set.C
papi_best_set_CXXFLAGS  = -I$(COMMON_INC) @PAPI_CFLAGS@
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


################
### pyextrae ###
################

EXTRA_DIST = \
  pyextrae/__init__.py \
	pyextrae/cuda/__init__.py \
  pyextrae/mpi/__init__.py \
  pyextrae/multiprocessing/__init__.py \
	pyextrae/profile/__init__.py \
  pyextrae/sequential/__init__.py \
  pyextrae/common/__init__.py \
  pyextrae/common/extrae.py.in

install-data-hook: 
	cp pyextrae/common/extrae.py.in pyextrae/common/extrae.py
	$(top_srcdir)/substitute $(SED) "@sub_PREFIX@" ${prefix} pyextrae/common/extrae.py
#if HAVE_PAPI
	$(top_srcdir)/substitute $(SED) "@sub_Extrae_eventandcounters@" Extrae_eventandcounters pyextrae/common/extrae.py
#else
	$(top_srcdir)/substitute $(SED) "@sub_Extrae_eventandcounters@" Extrae_event pyextrae/common/extrae.py
#endif
	mkdir -p $(libexecdir)/pyextrae
	cp -rL pyextrae/* $(libexecdir)/pyextrae
	rm $(libexecdir)/pyextrae/common/extrae.py.in

CLEANFILES = pyextrae/common/extrae.py

