/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2002 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* This file implements support for pruned MAC addresses. */

#ifndef _gm_global_id_h_
#define _gm_global_id_h_

#include "gm_global_id.h"

#define GM_MAC_VENDOR_MYRI 0x0060dd
#define GM_MAC_VENDOR_BYTE_MYRI 0xdd

/* GM node IDs are 32-bit compressed versions of 48-bit ethernet MAC
   addrs.  The last 3 bytes of each are the same, but the GM node ID
   uses the first byte as a vendor ID, where the MAC addr uses the
   first 3 bytes as a vendor ID.  (All this is from the big-endian
   perspective.) */
   
static inline
gm_status_t
gm_mac_addr_to_global_id (const gm_u8_t *mac_addr, gm_u32_t *global_id)
{
  unsigned int mac_vendor;
  gm_u8_t high_byte;

  /* Extract the high 3 bytes of the MAC addr, the "MAC vendor ID" */
  
  mac_vendor = (mac_addr[0] << 16) | (mac_addr[1] << 8) | mac_addr[2];

  /* Use the mac vendor to determine the high byte of the node ID. */
  
  switch (mac_vendor)
    {
    case GM_MAC_VENDOR_MYRI:
      high_byte = GM_MAC_VENDOR_BYTE_MYRI;
      break;
    default:
      return GM_INVALID_PARAMETER;
    }

  /* Build and return the node ID. */
  
  *global_id = ((high_byte << 24)
	      | (mac_addr[3] << 16)
	      | (mac_addr[4] << 8)
	      | mac_addr[5]);
  return GM_SUCCESS;
}

static inline
gm_status_t
gm_global_id_to_mac_addr (gm_u32_t global_id, gm_u8_t *mac_addr)
{
  gm_u8_t vendor;
  
  /* Extract the high byte of the vendor ID. */

  vendor = (gm_u8_t) ((global_id >> 24) & 0xff);

  /* Write the first 3 bytes of the MAC addr based on this vendor ID. */
  
  switch (vendor)
    {
    case GM_MAC_VENDOR_BYTE_MYRI:
      mac_addr[0] = (gm_u8_t) (GM_MAC_VENDOR_MYRI >> 16);
      mac_addr[1] = (gm_u8_t) (GM_MAC_VENDOR_MYRI >> 8);
      mac_addr[2] = (gm_u8_t) GM_MAC_VENDOR_MYRI;
      break;
    default:
      return GM_INVALID_PARAMETER;
    }
  mac_addr[3] = (gm_u8_t) (global_id >> 16);
  mac_addr[4] = (gm_u8_t) (global_id >> 8);
  mac_addr[5] = (gm_u8_t) global_id;
  return GM_SUCCESS;
}

#endif /* _gm_global_id_h_ */
