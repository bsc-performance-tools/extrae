
noinst_LTLIBRARIES = libonline.la

libonline_la_SOURCES  = \
  OnlineControl.cpp \
  Syncing.c         \
  OnlineConfig.cpp

libonline_la_CFLAGS   = `@MRNETAPP_CONFIG@ --fe-cflags` `@MRNETAPP_CONFIG@ --be-cflags` @MPI_CFLAGS@
libonline_la_CXXFLAGS = `@MRNETAPP_CONFIG@ --fe-cflags` `@MRNETAPP_CONFIG@ --be-cflags` 
libonline_la_LDFLAGS  = `@MRNETAPP_CONFIG@ --fe-libs`   `@MRNETAPP_CONFIG@ --be-libs`

