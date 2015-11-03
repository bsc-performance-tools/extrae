
include $(top_srcdir)/PATHS
include extractors/Makefile.inc

SUBDIRS = extractors

bin_PROGRAMS = online_root.bin

lib_LTLIBRARIES = 

noinst_LTLIBRARIES = libonline.la

SCRIPT_ONLINE_ENV  = extrae_online.sh
SCRIPT_ONLINE_ROOT = online_root
noinst_SCRIPTS     = $(SCRIPT_ONLINE_ENV) $(SCRIPT_ONLINE_ROOT)

EXTRA_DIST = $(SCRIPT_ONLINE_ENV).in $(SCRIPT_ONLINE_ROOT).in

#########################
###     BINARIES      ###
#########################

online_root_bin_SOURCES = \
  OnlineRoot.cpp     OnlineRoot.h       \
  OnlineConfig.cpp   OnlineConfig.h     \
  OnlineUtils.cpp    OnlineUtils.h      \
  Binder.cpp         Binder.h           \
  Messaging.cpp      Messaging.h        \
  xml-parse-online.c xml-parse-online.h \
  KnownProtocols.h
online_root_bin_CFLAGS   = \
  -I$(COMMON_INC) -I$(TRACER_INC) 
online_root_bin_CXXFLAGS = \
  -I$(COMMON_INC) -I$(TRACER_INC) \
  `@SYNAPSE_CONFIG@ --fe-cflags` $(CONTAINERS_INCLUDES)
online_root_bin_LDFLAGS = \
  `@SYNAPSE_CONFIG@ --fe-libs` `@SYNAPSE_CONFIG@ --rpath`

if HAVE_XML2
online_root_bin_CFLAGS   += @XML2_CFLAGS@
online_root_bin_CXXFLAGS += @XML2_CFLAGS@
online_root_bin_LDFLAGS  += -L@XML2_SHAREDLIBSDIR@ -R @XML2_SHAREDLIBSDIR@ @XML2_LIBS@
endif

if HAVE_SPECTRAL
online_root_bin_SOURCES += \
  SpectralRoot.cpp SpectralRoot.h
online_root_bin_CXXFLAGS += @SPECTRAL_CXXFLAGS@
online_root_bin_LDFLAGS  += @SPECTRAL_LDFLAGS@ @SPECTRAL_LIBS@
endif

if HAVE_CLUSTERING
online_root_bin_SOURCES += \
  ClusteringRoot.cpp ClusteringRoot.h
online_root_bin_CXXFLAGS += @CLUSTERING_CXXFLAGS@
online_root_bin_LDFLAGS  += @CLUSTERING_LDFLAGS@ @CLUSTERING_LIBS@
endif

online_root_bin_SOURCES += \
  GremlinsRoot.cpp GremlinsRoot.h

online_root_bin_LDADD        = extractors/libcontainers-fe.la $(STATS_LIB)/libmpistats.la $(COMMON_LIB)/libcommon.la
online_root_bin_DEPENDENCIES = $(STATS_LIB)/libmpistats.la


#########################
###     LIBRARIES     ###
#########################

libonline_la_SOURCES  = \
  OnlineControl.cpp  OnlineControl.h    \
  OnlineConfig.cpp   OnlineConfig.h     \
  OnlineUtils.cpp    OnlineUtils.h      \
  Binder.cpp         Binder.h           \
  Syncing.c          Syncing.h          \
  Messaging.cpp      Messaging.h        \
  xml-parse-online.c xml-parse-online.h \
  online_buffers.h   online_events.h    \
  filter_defs.h      tags.h             \
  KnownProtocols.h

libonline_la_CFLAGS    = -I$(COMMON_INC) -I$(TRACER_INC) @MPI_CFLAGS@

libonline_la_CXXFLAGS  = \
  -I$(COMMON_INC) -I$(INCLUDE_DIR) -I$(CLOCKS_INC) -I$(HWC_INC) -I$(TRACER_INC) -I$(BUFFERS_INC) \
  $(EXTRACTORS_INCLUDES) \
  `@SYNAPSE_CONFIG@ --be-cflags`

libonline_la_LDFLAGS   = \
  `@SYNAPSE_CONFIG@ --be-libs` `@SYNAPSE_CONFIG@ --rpath`

if HAVE_XML2
libonline_la_CFLAGS += @XML2_CFLAGS@
endif

if HAVE_SPECTRAL
libonline_la_SOURCES += \
  SpectralWorker.cpp SpectralWorker.h

libonline_la_CXXFLAGS += @SPECTRAL_CXXFLAGS@
libonline_la_LDFLAGS  += @SPECTRAL_LDFLAGS@ @SPECTRAL_LIBS@
endif

if HAVE_CLUSTERING
libonline_la_SOURCES += \
  ClusteringWorker.cpp ClusteringWorker.h

libonline_la_CXXFLAGS += @CLUSTERING_CXXFLAGS@
libonline_la_LDFLAGS  += @CLUSTERING_LDFLAGS@ @CLUSTERING_LIBS@
endif

libonline_la_SOURCES += \
  GremlinsWorker.cpp GremlinsWorker.h
libonline_la_CXXFLAGS += @MPI_CFLAGS@

libonline_la_LIBADD = extractors/libextractors.la extractors/libcontainers-be.la


#########################
###      FILTERS      ###
#########################

if HAVE_SPECTRAL
lib_LTLIBRARIES += libfilterOnlineSpectral.la

libfilterOnlineSpectral_la_SOURCES  = \
  SpectralFilter.cpp SpectralFilter.h 
libfilterOnlineSpectral_la_CXXFLAGS = `@SYNAPSE_CONFIG@ --cp-cflags` $(CONTAINERS_INCLUDES) @SPECTRAL_CXXFLAGS@
libfilterOnlineSpectral_la_LDFLAGS  = @SPECTRAL_LDFLAGS@ @SPECTRAL_LIBS@
libfilterOnlineSpectral_la_LIBADD   = extractors/libcontainers-cp.la
endif


#########################
###      SCRIPTS      ###
#########################

$(SCRIPT_ONLINE_ENV): $(SCRIPT_ONLINE_ENV).in
	cp $^ $@
	$(top_srcdir)/substitute $(SED) "@sub_PREFIX@" ${prefix} $@
	$(top_srcdir)/substitute $(SED) "@sub_SYNAPSE_HOME@" @SYNAPSE_HOME@ $@
	$(top_srcdir)/substitute $(SED) "@sub_SYNAPSE_LIBSDIR@" @SYNAPSE_LIBSDIR@ $@
	$(top_srcdir)/substitute $(SED) "@sub_CLUSTERING_HOME@" @CLUSTERING_HOME@ $@

$(SCRIPT_ONLINE_ROOT): $(SCRIPT_ONLINE_ROOT).in
	cp $^ $@
	$(top_srcdir)/substitute $(SED) "@sub_PREFIX@" ${prefix} $@
	chmod u+x $@

.PHONY : $(SCRIPT_ONLINE_ENV) $(SCRIPT_ONLINE_ROOT)

install-data-hook:
	$(MKDIR_P) $(prefix)/etc
	cp $(SCRIPT_ONLINE_ENV) $(prefix)/etc
	cp $(SCRIPT_ONLINE_ROOT) $(prefix)/bin
	chmod a+x $(prefix)/bin/$(SCRIPT_ONLINE_ROOT)

clean-local:
	rm -f $(SCRIPT_ONLINE_ENV) $(SCRIPT_ONLINE_ROOT)

