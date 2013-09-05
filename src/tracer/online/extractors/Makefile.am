
include $(top_srcdir)/PATHS

noinst_LTLIBRARIES = libextractors.la libcontainers.la

EXTRACTORS_SRCS  = \
  BufferExtractor.cpp BufferExtractor.h \
  BurstsExtractor.cpp BurstsExtractor.h \
  Chopper.cpp Chopper.h

CONTAINERS_SRCS  = \
  Bursts.cpp Bursts.h 
if HAVE_SPECTRAL
CONTAINERS_SRCS += \
  Signal.cpp Signal.h
endif 

libextractors_la_SOURCES = $(EXTRACTORS_SRCS) 
libcontainers_la_SOURCES = $(CONTAINERS_SRCS)

libextractors_la_CXXFLAGS  = -I. -I$(COMMON_DIR) -I$(BUFFERS_DIR) -I$(TRACER_DIR) -I$(CLOCKS_DIR) -I$(HWC_DIR) -I$(ONLINE_DIR) `@MRNAPP_CONFIG@ --be-cflags`
libcontainers_la_CXXFLAGS  = -I. -I$(COMMON_DIR) -I$(ONLINE_DIR) -I$(HWC_DIR) `@MRNAPP_CONFIG@ --cp-cflags`

if HAVE_SPECTRAL
libextractors_la_CXXFLAGS += @SPECTRAL_CXXFLAGS@
libcontainers_la_CXXFLAGS += @SPECTRAL_CXXFLAGS@
endif

