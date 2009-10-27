/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

#ifndef _gm_page_table_h_
#define _gm_page_table_h_

/****************************************************************
 * Switches
 ****************************************************************/

#define GM_DEBUG_PAGE_HASH 0
#define GM_DEBUG_HASH 0

/****************************************************************
 * Includes
 ****************************************************************/

#include "gm.h"
#include "gm_call_trace.h"
#include "gm_crc32.h"
#include "gm_debug.h"

/****************************************************************
 * Globals
 ****************************************************************/

/****************************************************************
 * Macros and inlines
 ****************************************************************/

static gm_inline
unsigned
GM_HASH_PAGE_PORT (gm_up_t page_port)
{
  gm_u32_t c;		/* accumulated crc32 */

  GM_CALLED_WITH_ARGS_EX
    (GM_BUILDING_FIRMWARE && GM_DEBUG_HASH,
     ("GM_HASH_PAGE_PORT (0x%qx) called\n", (gm_u64_t) page_port));

#if GM_BUILDING_FIRMWARE
  /* It is safe to use the global hardware implementation (or the
     software emulation of it) to compute the CRC-32. */
  GM_CRC32_SET (0xffffffffUL);
  if (sizeof (gm_up_t) == 8)
    {
      GM_CRC32_WORD ((gm_u32_t) (page_port >> (8 * sizeof (gm_up_t) / 2)));
    }
  GM_CRC32_WORD ((gm_u32_t) page_port);
  GM_CRC32_WORD (0);
  c = GM_CRC32_GET ();
#else
  /* In the host, we use the software implementation that is confirmed to
     agree with the hardware (and the AAL-5 test vectors). */
  c = 0xffffffffUL;
  if (sizeof (gm_up_t) == 8)
    {
      gm_crc32_u32 ((gm_u32_t) (page_port >> (8 * sizeof (gm_up_t) / 2)), &c);
    }
  gm_crc32_u32 ((gm_u32_t) page_port, &c);
  gm_crc32_u32 ((gm_u32_t) 0, &c);
#endif
  /* We skip the useless (to us) "c = ~c" CRC step. */
  
#if GM_BUILDING_FIRMWARE
  GM_PRINT
    (GM_DEBUG_HASH,
     ("page_port 0x%qx hashes to 0x%x\n",
      (gm_u64_t) page_port, (unsigned int) c));
#else
  GM_PRINT (GM_DEBUG_HASH,
         ("page_port 0x%lx hashes to 0x%x\n",
          (unsigned long) page_port, (unsigned int) c));
#endif

  GM_RETURN (c);
}

/****************************************************************
 * function prototypes
 ****************************************************************/

/****************
 * kernel
 ****************/

#if GM_KERNEL
gm_dp_t gm_dma_addr_for_mapping (gm_instance_state_t * is, gm_up_t user_vma,
				 gm_u32_t port);
#endif /* GM_KERNEL */


#endif /* _gm_page_table_h_ */
