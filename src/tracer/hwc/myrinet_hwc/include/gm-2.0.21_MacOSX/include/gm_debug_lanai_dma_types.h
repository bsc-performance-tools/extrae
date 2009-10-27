/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1998 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_debug_lanai_dma_types_h_
#define _gm_debug_lanai_dma_types_h_

/* Set this to use Loic's bitmap-based DMA addr checking to prevent
   DMAs to pages not registered for GM DMA. */
#define GM_DEBUG_LANAI_DMA 0

/* Set this to print details of all DMA page registrations and all
   DMAs as they start. */
#define GM_TRACE_LANAI_DMA 0

/****************************************************************/

#include "gm_simple_types.h"
#include "gm_bitmap.h"

/* This is used to size the dma_pages_bitmap, which is used iff
   GM_DEBUG_LANAI_DMA is set, so set this to 1 to minimize memory
   wastage when this GM_DEBUG_LANAI_DMA feature is not used.

   (1<<21) is large enough to check the bottom 33 bits on all archs. */

#define GM_MAX_HOST_DMA_PAGES	(GM_DEBUG_LANAI_DMA ? (1<<21) : 1)

typedef GM_BITMAP_DECL(gm_dma_page_bitmap_t, GM_MAX_HOST_DMA_PAGES);

gm_status_t __gm_noticed_dma_addr (gm_dma_page_bitmap_t *bitmap, gm_dp_t dp);
void __gm_notice_dma_addr (gm_dma_page_bitmap_t *bitmap, gm_dp_t dp);
void __gm_forget_dma_addr (gm_dma_page_bitmap_t *bitmap, gm_dp_t dp);

#endif /* _gm_debug_lanai_dma_types_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
