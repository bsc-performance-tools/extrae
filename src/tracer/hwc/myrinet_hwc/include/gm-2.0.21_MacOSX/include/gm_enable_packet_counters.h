/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* This file includes support for packet counters, which slow down the
   critical path of the firmware. */

#ifndef _gm_enable_packet_counters_h_
#define _gm_enable_packet_counters_h_

/* please leave this on for any source release to customers - it helps
   debugging */
#define GM_ENABLE_PACKET_COUNTERS 1

/* Handy related macros. These take the old value as a parameter to allow
   for better pipelining in the calling code. */

#if GM_ENABLE_PACKET_COUNTERS
#define GM_INCR_PACKET_CNT(c, old_c) do {				\
  gm_assert ((c) == (old_c));						\
  (c) = (old_c) + 1;							\
} while (0);
#define GM_DECR_PACKET_CNT(c) do {					\
  gm_assert ((c) == (old_c));						\
  (c) = (old_c) - 1;							\
} while (0);
#else
#define GM_INCR_PACKET_CNT(c, old_c) GM_VAR_MAY_BE_UNUSED (old_c)
#define GM_DECR_PACKET_CNT(c, old_c) GM_VAR_MAY_BE_UNUSED (old_c)
#endif

#endif /* _gm_enable_packet_counters_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
