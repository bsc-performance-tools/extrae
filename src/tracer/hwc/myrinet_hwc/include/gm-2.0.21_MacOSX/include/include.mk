This file is obsolete.

#############
# Directories
#############

include_srcdir = $(top_srcdir)/include

#############
# Headers (and the headers they depend upon)
#############

gm_h = $(include_srcdir)/gm.h
gm_auto_config_h = $(include_builddir)/gm_auto_config.h
gm_nt_config_h = $(include_srcdir)/gm_nt_config.h
gm_lanai3_config_h = $(include_srcdir)/gm_lanai3_config.h
gm_config_h = $(include_srcdir)/gm_config.h $(gm_nt_config_h) \
	$(gm_lanai3_config_h) $(gm_auto_config_h)
gm_types_h = $(include_srcdir)/gm_types.h $(gm_h) $(gm_config_h)
gm_lanai_h = $(include_srcdir)/gm_lanai.h $(gm_types_h)
gm_debug_h = $(include_srcdir)/gm_debug.h $(gm_config_h)
gm_io_h = $(include_srcdir)/gm_io.h $(gm_types_h)
gm_zone_types_h = $(include_srcdir)/gm_zone_types.h $(gm_types_h)
gm_zone_h = $(include_srcdir)/gm_zone.h $(gm_zone_types_h)
gm_internal_h = $(include_srcdir)/gm_internal.h $(gm_h) $(gm_types_h) \
	$(gm_lanai_h) $(gm_io_h) $(gm_zone_types_h) $(gm_debug_h) \
	$(gm_zone_h)
gm_page_hash_h = $(include_srcdir)/gm_page_hash.h

myriProm_h = $(include_srcdir)/myriProm.h
myriInterface_h = $(include_srcdir)/myriInterface.h
lanai4_def_h = $(include_srcdir)/lanai4_def.h
lanai5_def_h = $(include_srcdir)/lanai5_def.h
lanai6_def_h = $(include_srcdir)/lanai6_def.h
gm_lanai_def_h = $(include_srcdir)/gm_lanai_def.h