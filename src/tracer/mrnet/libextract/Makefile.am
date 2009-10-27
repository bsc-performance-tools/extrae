include $(top_srcdir)/PATHS

COMMON_SRCS = BurstInfo.C
FE_SRCS = BurstInfo_FE.C
BE_SRCS = BurstInfo_BE.C LongGlops.C

noinst_LTLIBRARIES = libextract_fe.la libextract_be.la

libextract_fe_la_SOURCES = $(FE_SRCS) $(COMMON_SRCS)
libextract_fe_la_CXXFLAGS = -I. -I$(COMMON_DIR) $(CLUSTERS_INCS) $(SPECTRAL_CXXFLAGS)
libextract_fe_la_CFLAGS = -I. -I$(COMMON_DIR)

libextract_be_la_SOURCES = $(BE_SRCS) $(COMMON_SRCS)
libextract_be_la_CXXFLAGS = -I. -I$(COMMON_DIR) -I$(WRAPPERS_DIR)/API -I$(TRACER_DIR) -I$(HWC_DIR) $(SPECTRAL_CXXFLAGS) @MPI_CFLAGS@

