/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1998 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_debug_lanai_dma_h_
#define _gm_debug_lanai_dma_h_

#include "gm_debug_lanai_dma_types.h"

static gm_inline gm_status_t
gm_noticed_dma_addr (gm_dma_page_bitmap_t *bitmap, gm_dp_t dp)
{
  gm_assert (bitmap);
  if (GM_DEBUG_LANAI_DMA)
    {
      return __gm_noticed_dma_addr (bitmap, dp);
    }
  else
    {
      return GM_SUCCESS;
    }
}

static gm_inline void
_gm_notice_dma_addr (gm_dma_page_bitmap_t *bitmap, gm_dp_t dp)
{
  if (GM_DEBUG_LANAI_DMA)
    {
      __gm_notice_dma_addr (bitmap, dp);
    }
}

#define gm_notice_dma_addr(bitmap, dp) do {	\
  gm_assert (bitmap);				\
  _gm_notice_dma_addr (bitmap, dp);		\
} while (0)

static gm_inline void
_gm_forget_dma_addr (gm_dma_page_bitmap_t *bitmap, gm_dp_t dp)
{
  if (GM_DEBUG_LANAI_DMA)
    {
      __gm_forget_dma_addr (bitmap, dp);
    }
}

#define gm_forget_dma_addr(bitmap, dp) do {	\
  gm_assert (bitmap);				\
  _gm_forget_dma_addr (bitmap, dp);		\
} while (0)

#endif /* _gm_debug_lanai_dma_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
