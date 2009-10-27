include $(top_srcdir)/PATHS

SUBDIRS = libextract

noinst_LTLIBRARIES = libmrn.la
libmrn_la_SOURCES  = mrnet_be.C mrnet_be.h mrnet_commands.h be_protocol.C be_protocol.h protocol.C protocol.h StreamPublisher.C mrn_config.cpp BE_ProtScope.cpp
libmrn_la_CXXFLAGS = -I$(COMMON_DIR) -I$(TRACER_DIR) -I$(CLOCKS_DIR) -I$(HWC_DIR) -I$(WRAPPERS_DIR)/API @MRNET_CXXFLAGS@ @SPECTRAL_CXXFLAGS@ -DBE
if HAVE_MPI
libmrn_la_CXXFLAGS += @MPI_CXXFLAGS@
endif
libmrn_la_LIBADD = libextract/libextract_be.la

bin_PROGRAMS = tracemon mrnet_root clusterize spectralize
mrnet_root_SOURCES  = protocol.C root_protocol.C mrnet_root.C commands_queue.C BEMask.C StreamPublisher.C mrn_config.cpp SpectralTool.cpp FE_ProtScope.cpp
mrnet_root_CXXFLAGS = -g -I$(COMMON_DIR) -Ilibextract @MRNET_CXXFLAGS@ @CLUSTERING_CXXFLAGS@ @SPECTRAL_CXXFLAGS@ -DFE
mrnet_root_LDFLAGS  = -L@MRNET_LIBSDIR@ -R @MRNET_LIBSDIR@ @MRNET_LIBS@ @CLUSTERING_LDFLAGS@ @CLUSTERING_LIBS@ @SPECTRAL_LDFLAGS@ @SPECTRAL_LIBS@ -Wl,-E 
mrnet_root_LDADD    = $(COMMON_DIR)/libcommon.la libextract/libextract_fe.la

tracemon_SOURCES = tracemon.c tracemon.h
tracemon_CFLAGS = -I$(COMMON_DIR)
tracemon_LDFLAGS = -lpthread

clusterize_SOURCES = clusterize.C 
clusterize_CXXFLAGS = -I$(COMMON_DIR) -Ilibextract @CLUSTERING_CXXFLAGS@
clusterize_LDFLAGS = @CLUSTERING_LDFLAGS@ @CLUSTERING_LIBS@
clusterize_LDADD = $(COMMON_DIR)/libcommon.la libextract/libextract_fe.la

spectralize_SOURCES = spectralize.C SpectralTool.cpp
spectralize_CXXFLAGS = -I$(COMMON_DIR) -Ilibextract @SPECTRAL_CXXFLAGS@
spectralize_LDFLAGS = @SPECTRAL_LDFLAGS@ @SPECTRAL_LIBS@
spectralize_LDADD =  $(COMMON_DIR)/libcommon.la libextract/libextract_fe.la

all: mrn_filters.so
mrn_filters.so: mrnet_filters.C mrnet_commands.h
	$(CXX) $(CXXFLAGS) -I$(BASE_DIR) -I$(COMMON_DIR) -fPIC -shared -rdynamic @MRNET_CXXFLAGS@ -o $@ $< 

install-data-hook:
	cp mrn_filters.so $(prefix)/bin

