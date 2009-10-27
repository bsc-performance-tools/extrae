/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

#ifndef _gm_zone_h_
#define _gm_zone_h_

#include "gm_zone_types.h"

GM_ENTRY_POINT struct gm_zone *
gm_zone_create_zone (struct gm_zone *zone, void *base, gm_size_t length);

GM_ENTRY_POINT void
gm_zone_destroy_zone (struct gm_zone *zone);

GM_ENTRY_POINT void *
gm_zone_free (struct gm_zone *zone, void *a);

GM_ENTRY_POINT void *
gm_zone_malloc (struct gm_zone *zone, gm_size_t length);

GM_ENTRY_POINT void *
gm_zone_calloc (struct gm_zone *zone, gm_size_t count, gm_size_t length);

GM_ENTRY_POINT void *
gm_zone_bzero (void *s_pv, gm_size_t c);

GM_ENTRY_POINT int
gm_zone_addr_in_zone (struct gm_zone *zone, void *p);

#endif
