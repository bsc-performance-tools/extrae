
include $(top_srcdir)/PATHS

SUBDIRS = extractors

lib_LTLIBRARIES    = 

noinst_LTLIBRARIES = libonline.la

ONLINE_ENV_SH      = online_env.sh
noinst_SCRIPTS     = $(ONLINE_ENV_SH)

#########################
###     LIBRARIES     ###
#########################

libonline_la_SOURCES  =               \
  OnlineControl.cpp  OnlineControl.h  \
  Syncing.c          Syncing.h        \
  OnlineConfig.cpp   OnlineConfig.h   \
  SpectralRoot.cpp   SpectralRoot.h   \
  SpectralWorker.cpp SpectralWorker.h \
  filter_defs.h      online_buffers.h \
  online_events.h    tags.h

EXTRA_DIST = \
  $(ONLINE_ENV_SH).in
  

# DEBUG
#  mrnet_threads.cpp \
#  Syncing.c         \
#  OnlineConfig.cpp

EXTRACTORS_INCLUDES = -I$(ONLINE_DIR)/extractors -I$(COMMON_DIR) -I$(BUFFERS_DIR) 
CONTAINERS_INCLUDES = -I$(ONLINE_DIR)/extractors -I$(COMMON_DIR)

libonline_la_CFLAGS    = -I$(COMMON_DIR) @MPI_CFLAGS@

libonline_la_CXXFLAGS  = \
  -I$(COMMON_DIR) -I$(INCLUDE_DIR) -I$(CLOCKS_DIR) -I$(HWC_DIR) -I$(TRACER_DIR) -I$(BUFFERS_DIR) \
  $(EXTRACTORS_INCLUDES) \
  `@MRNETAPP_CONFIG@ --fe-cflags` \
  `@MRNETAPP_CONFIG@ --be-cflags`

libonline_la_LDFLAGS   = \
  `@MRNETAPP_CONFIG@ --fe-libs` \
  `@MRNETAPP_CONFIG@ --be-libs`

if HAVE_SPECTRAL
libonline_la_CXXFLAGS += @SPECTRAL_CXXFLAGS@
libonline_la_LDFLAGS  += @SPECTRAL_LDFLAGS@ @SPECTRAL_LIBS@
endif
libonline_la_LIBADD    = extractors/libextractors.la


#########################
###      FILTERS      ###
#########################

if HAVE_SPECTRAL
lib_LTLIBRARIES += libfilterOnlineSpectral.la

libfilterOnlineSpectral_la_SOURCES  = \
  SpectralFilter.cpp SpectralFilter.h 
libfilterOnlineSpectral_la_CXXFLAGS = `@MRNETAPP_CONFIG@ --cp-cflags` $(CONTAINERS_INCLUDES) @SPECTRAL_CXXFLAGS@
libfilterOnlineSpectral_la_LDFLAGS  = @SPECTRAL_LDFLAGS@ @SPECTRAL_LIBS@
libfilterOnlineSpectral_la_LIBADD   = extractors/libcontainers.la
endif


#########################
###      SCRIPTS      ###
#########################

$(ONLINE_ENV_SH): $(ONLINE_ENV_SH).in
	cp $^ $@
	$(top_srcdir)/substitute $(SED) "@sub_PREFIX@" ${prefix} $@
	$(top_srcdir)/substitute $(SED) "@sub_MRNETAPP_HOME@" @MRNETAPP_HOME@ $@
	$(top_srcdir)/substitute $(SED) "@sub_MRNETAPP_LIBSDIR@" @MRNETAPP_LIBSDIR@ $@

install-data-hook:
	$(mkdir_p) $(prefix)/etc
	cp $(ONLINE_ENV_SH) $(prefix)/etc

clean-local:
	rm -f $(ONLINE_ENV_SH)

