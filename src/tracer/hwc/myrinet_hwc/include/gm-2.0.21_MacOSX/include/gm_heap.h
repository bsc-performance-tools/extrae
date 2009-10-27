/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2004 by Myricom, Inc.					 *
 * All rights reserved.  See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_heap_h_
#define _gm_heap_h_

#include "gm.h"

struct gm_heap;

typedef gm_offset_t gm_heap_compare_func_t (const void *key1,
					    const void *key2);

GM_ENTRY_POINT gm_status_t gm_heap_create (struct gm_heap **_h,
					   gm_heap_compare_func_t *compare,
					   gm_size_t min_entries);
GM_ENTRY_POINT void gm_heap_destroy (struct gm_heap *h);
GM_ENTRY_POINT gm_status_t gm_heap_insert (struct gm_heap *h, void *key);
GM_ENTRY_POINT void *gm_heap_peek (struct gm_heap *h);
GM_ENTRY_POINT void *gm_heap_remove (struct gm_heap *h);
GM_ENTRY_POINT void gm_heap_sort (void **array,
				  gm_size_t cnt,
				  gm_heap_compare_func_t *compare);

#endif /* _gm_heap_h_ */
