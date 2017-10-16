include $(top_srcdir)/PATHS

noinst_LTLIBRARIES = libintel-kmpc-11-intermediate.la

libintel_kmpc_11_intermediate_la_SOURCES = intel-kmpc-11-intermediate-part1.c intel-kmpc-11-intermediate-part2.c intel-kmpc-11-intermediate-part3.c intel-kmpc-11-intermediate-part4.c intel-kmpc-11-intermediate-part5.c  intel-kmpc-11-taskloop-helpers.c intel-kmpc-11-taskloop-helpers.h
libintel_kmpc_11_intermediate_la_CFLAGS = -Wno-unused-function -I$(INCLUDE_DIR) -I$(WRAPPERS_DIR)/API -I$(WRAPPERS_DIR)/OMP -I$(CLOCKS_DIR) -I$(TRACER_DIR) -I$(COMMON_DIR) -I$(HWC_DIR)

EXTRA_DIST = $(libintel_kmpc_11_intermediate_la_SOURCES)
