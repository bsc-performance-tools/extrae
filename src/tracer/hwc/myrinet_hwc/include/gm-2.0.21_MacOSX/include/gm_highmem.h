/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2004 by Myricom, Inc.					 *
 * All rights reserved.  See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* Compute the layout of high memory. */

#ifndef _gm_highmem_h_
#define _gm_highmem_h_

#include "gm.h"
#include "gm_types.h"

static
gm_status_t
gm_highmem_compute_layout (unsigned long available_memory,
			   unsigned long *_connection_cnt,
			   unsigned long *_cached_pte_cnt,
			   unsigned long *_bitmap_len,
			   const unsigned long max_connection_cnt)
{
  unsigned long connection_cnt;
  unsigned long cached_pte_cnt;
  unsigned long bitmap_len;

  /* Compute a preliminary connection count using up to 1/2 of the
     available memory, but not exceeding max_connection_cnt. */
  
  connection_cnt = available_memory / 2 / sizeof (gm_connection_t);
  if (connection_cnt > max_connection_cnt)
    {
      connection_cnt = max_connection_cnt;
    }
  available_memory -= connection_cnt * sizeof (gm_connection_t);

  /* Compute the largest power-of-two number of PTEs that will fit
     in the remaining memory. */
  
  cached_pte_cnt = available_memory / sizeof (gm_cached_pte_t) - 1;
  if (!GM_POWER_OF_TWO (cached_pte_cnt))
    {
      cached_pte_cnt = 1 << (gm_log2_roundup (cached_pte_cnt) - 1);
    }
  /* Reserve space for cached PTEs. */
  available_memory -= (cached_pte_cnt + 1) * sizeof (gm_cached_pte_t);

  /* Compute the size of the PTE cache entry bitmap. */
  
  bitmap_len = GM_ROUNDUP (u32, (cached_pte_cnt + 1) / 8 + 1, 8);

  /* Recompute the number of connections, using all memory not
     used for the PTE cache and its bitmap. */

  /* Release the preliminary allocation. */
  available_memory += connection_cnt * sizeof (gm_connection_t);
  /* reserve the bitmap memory.  This is not done earlier to avoid possible
     unsigned underflow, which is handled poorly on x86. */
  available_memory -= bitmap_len;
  /* compute the new number of connections, checking for overflow. */
  connection_cnt = available_memory / sizeof (gm_connection_t);
  if (connection_cnt > max_connection_cnt)
    {
      connection_cnt = max_connection_cnt;
    }

  /* Return computed layout details. */
  
  *_connection_cnt = connection_cnt;
  *_cached_pte_cnt = cached_pte_cnt;
  *_bitmap_len = bitmap_len;
  return GM_SUCCESS;
}



#endif /* _gm_highmem_h_ */
