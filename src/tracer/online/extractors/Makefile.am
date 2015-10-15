
include $(top_srcdir)/PATHS
include Makefile.inc

noinst_LTLIBRARIES = libextractors.la libcontainers-fe.la libcontainers-cp.la libcontainers-be.la

EXTRACTORS_SRCS  = \
  BufferParser.cpp    BufferParser.h    \
  BurstsExtractor.cpp BurstsExtractor.h \
  Chopper.cpp         Chopper.h

CONTAINERS_SRCS  = \
  Bursts.cpp     Bursts.h \
  PhaseStats.cpp PhaseStats.h
if HAVE_SPECTRAL
CONTAINERS_SRCS += \
  Signal.cpp Signal.h
endif 

libextractors_la_SOURCES  = $(EXTRACTORS_SRCS) 
libextractors_la_CXXFLAGS = $(EXTRACTORS_INCLUDES)

libcontainers_fe_la_SOURCES  = $(CONTAINERS_SRCS)
libcontainers_fe_la_CXXFLAGS = $(CONTAINERS_INCLUDES) `@SYNAPSE_CONFIG@ --fe-cflags`

libcontainers_cp_la_SOURCES  = $(CONTAINERS_SRCS)
libcontainers_cp_la_CXXFLAGS = $(CONTAINERS_INCLUDES) `@SYNAPSE_CONFIG@ --cp-cflags`

libcontainers_be_la_SOURCES  = $(CONTAINERS_SRCS)
libcontainers_be_la_CXXFLAGS = $(CONTAINERS_INCLUDES) `@SYNAPSE_CONFIG@ --be-cflags`
