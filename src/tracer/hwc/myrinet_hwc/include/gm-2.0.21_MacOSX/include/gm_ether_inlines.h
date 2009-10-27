/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2003 by Myricom, Inc.					 *
 * All rights reserved.  See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_ether_inlines_h_
#define _gm_ether_inlines_h_

#include "gm_crc32.h"
#include "gm_ether.h"
#include "gm_types.h"

/****************************************************************
 * Ethernet MAC address manipulation.
 *
 * These functions assume MAC addresses are 2-byte aligned for
 * performance.
 ****************************************************************/

/* Compute a 32-bit hash given a MAC address.  ~10 instructions on LANaiX */

GM_FUNCTION_MAY_BE_UNUSED
static
gm_u32_t
gm_ethernet_mac_addr_hash (gm_ethernet_mac_addr_t mac)
{
  gm_u16_t *mac16, a, b, c;

  gm_assert (mac);
  
  mac16 = (gm_u16_t *)mac;
  gm_assert (GM_NATURALLY_ALIGNED (mac16));
    
  GM_CRC32_SET (0xffffffff);
  a = mac16[0];
  b = mac16[1];
  c = mac16[2];
  GM_CRC32_HALF (a);
  GM_CRC32_HALF (b);
  GM_CRC32_HALF (c);
  return GM_CRC32_GET ();
}

/* Test 2 macs for differences.  Return 0 if and only iff the MACs are
   identical.  (11 instructions) */

GM_FUNCTION_MAY_BE_UNUSED
static
gm_u32_t
gm_ethernet_mac_addrs_differ (gm_ethernet_mac_addr_t a,
			      gm_ethernet_mac_addr_t b)
{
  gm_u16_t *a16, *b16;
  
  gm_assert (a);
  gm_assert (b);
  
  a16 = (gm_u16_t *)a;
  gm_assert (GM_NATURALLY_ALIGNED (a16));

  b16 = (gm_u16_t *)b;
  gm_assert (GM_NATURALLY_ALIGNED (b16));

  return ((a16[0] - b16[0]) | (a16[1] - b16[1]) | (a16[2] - b16[2]));
}

/* Copy a MAC address, assuming the source and and destination are
   naturally aligned. (6 instructions) */

GM_FUNCTION_MAY_BE_UNUSED
static
void
gm_ethernet_mac_addr_copy (gm_ethernet_mac_addr_t _from,
			   gm_ethernet_mac_addr_t _to)
{
  gm_u16_t a, b, c;
  gm_u16_t *from, *to;

  gm_assert (_from);
  gm_assert (_to);
  
  from = (gm_u16_t *) _from;
  gm_assert (GM_NATURALLY_ALIGNED (from));
  to = (gm_u16_t *) _to;
  gm_assert (GM_NATURALLY_ALIGNED (to));

  /* Pipelined copy */
  
  a = *from++;
  b = *from++;
  c = *from++;
  *to++ = a;
  *to++ = b;
  *to++ = c;
}
#endif /* _gm_ether_inlines_h_ */
