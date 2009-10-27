/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2002 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

#ifndef _gm_dma_perf_h
#define _gm_dma_perf_h

#include "gm_config.h"

/* The number of pages used for dma performance testing when an
   mcp is first loaded.  gm_debug displays the performance results */

#define GM_NUM_DMA_TEST_SEGMENTS (GM_ENABLE_SPARC_STREAMING && \
  (GM_CPU_sparc || GM_CPU_sparc64) ? 16 : 32)

/* Determines whether or not to do the 'alternate' timings */
#define GM_DMA_PERF_ALTERNATIVE_TIMING 0

#endif /* _gm_dma_perf_h */
