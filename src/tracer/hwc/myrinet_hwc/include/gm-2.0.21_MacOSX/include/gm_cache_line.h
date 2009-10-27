/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2002 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_cache_line_h_
#define _gm_cache_line_h_

#include "gm.h"

/* What size cache line should we assume?  This may not be the actual
   cache line length, but is a reasonable assumption for code that
   attempts to consider caching effects of memory accesses.  It should
   be a power of two and as large as possible without being larger
   than the actual cache line size on any machine running the code.
   
   You may add more accurate cache line estimates if you know how
   to determine them at compile-time, but you may have to modify
   gm_crc32.h if you do. */

#if GM_SIZEOF_VOID_P == 8
#define GM_CACHE_LINE_LEN 128 /* bytes */
#elif GM_SIZEOF_VOID_P == 4
#define GM_CACHE_LINE_LEN 32 /* bytes */
#else
#error
#endif

/* Set this when the processor has no cache. */

#define GM_NO_CACHE GM_BUILDING_FIRMWARE

#endif /* _gm_cache_line_h_ */
