/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* This file includes support for debug counters, which slow down the
   critical path of the firmware. */

#ifndef _gm_enable_debug_counters_h_
#define _gm_enable_debug_counters_h_

#define GM_ENABLE_DEBUG_COUNTERS 0

/* automatic overrides */

/* Handy related macros */

#if GM_ENABLE_DEBUG_COUNTERS || GM_DEBUG
#define GM_INCR_DEBUG_CNT(c) (++(c ## _debug_cnt))
#define GM_DECR_DEBUG_CNT(c) (--(c ## _debug_cnt))
#else
#define GM_INCR_DEBUG_CNT(c)
#define GM_DECR_DEBUG_CNT(c)
#endif

#endif /* _gm_enable_debug_counters_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  c-backslash-column:72
  End:
*/
