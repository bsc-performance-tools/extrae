/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2001 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_debug_mpds_h_
#define _gm_debug_mpds_h_

#include "gm.h"

#define GM_DEBUG_MPDS 0

#if GM_DEBUG > GM_DEBUG_MPDS
#undef GM_DEBUG_MPDS
#define GM_DEBUG_MPDS GM_DEBUG
#endif

/* Special assert to allow MPD checking when debugging is off but
   GM_DEBUG_MPDS is set, as well as during conventional debugging.
   This is helpful to check for bugs that are not triggered when the
   firmware is slowed down by conventional debugging features. */

#if GM_DEBUG_MPDS >= 2
#define GM_MPD_ASSERT(a) _gm_always_assert (a, #a)
#elif GM_DEBUG_MPDS
#define GM_MPD_ASSERT(a) _gm_always_assert (a, 0)
#else
#define GM_MPD_ASSERT(a) /* Don't check syntax, since the debug fields
			    are optimized away when not debugging. */
#endif

#endif /* _gm_debug_mpds_h_ */
