/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2003 by Myricom, Inc.					 *
 * All rights reserved.  See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_gateway_h_
#define _gm_gateway_h_

#include "gm_ether.h"
#include "gm_simple_types.h"

/* The timeout (in LANai ticks) before invalidating cached gateway
   information. */
#define GM_GATEWAY_CACHE_TIMEOUT 300 /* seconds */

/****************************************************************
 * Function prototypes
 ****************************************************************/

#if GM_BUILDING_FIRMWARE
extern gm_u16_t gm_gateway (gm_ethernet_mac_addr_t destination);
extern void gm_notice_gateway (gm_u32_t global_id, gm_ethernet_mac_addr_t mac);
struct gm_gateway_cache_entry *entry_for_mac (gm_ethernet_mac_addr_t mac);
gm_status_t gm_gateway_cache_init (void);
void gm_debug_dump_gateway_cache (void);
#endif

#endif /* _gm_gateway_h_ */
