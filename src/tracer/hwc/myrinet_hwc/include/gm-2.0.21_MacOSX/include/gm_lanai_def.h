/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

#ifndef _gm_lanai_def_h_
#define _gm_lanai_def_h_

#define L4 0
#define L5 0
#define L6 0
#define L7 0
#define L8 0
#define L9 0
#define LX 0

/* HACK to convert a GM_LANAI_MAJOR_VERSION to a number */
#define X 10

#if GM_LANAI_MAJOR_VERSION == 4
#undef L4
#define L4 1
#elif GM_LANAI_MAJOR_VERSION == 5
#undef L5
#define L5 1
#elif GM_LANAI_MAJOR_VERSION == 6
#undef L6
#define L6 1
#elif GM_LANAI_MAJOR_VERSION == 7
#undef L7
#define L7 1
#elif GM_LANAI_MAJOR_VERSION == 8
#define L8 1
#elif GM_LANAI_MAJOR_VERSION == 9
#undef L9
#define L9 1
#elif GM_LANAI_MAJOR_VERSION == X
#undef LX
#define LX 1
#elif GM_LANAI_MAJOR_VERSION
#error GM_LANAI_MAJOR_VERSION not recognized
#else
#error GM_LANAI_MAJOR_VERSION not defined
#endif

/* Include the appropriate LANai definition header file */

#if L4
#include "lanai4_def.h"
#elif L5
#include "lanai5_def.h"
#elif L6
#include "lanai6_def.h"
#elif L7
#include "lanai7_def.h"
#elif L8
#include "lanai8_def.h"
#elif L9
#include "lanai9_def.h"
#elif LX
#include "lanaiX_def.h"
#endif

/* Define the fake DMA_INT_BIT for CPUs that don't define this already. */

#if L6 | L7
#define DMA_INT_BIT WAKE_INT_BIT
#elif L8 | L9
#define DMA_INT_BIT WAKE0_INT_BIT
#endif

/* Make sure we are using the correct compiler. */

#ifndef lanai			/* lanai-gcc OK for any lanai */
#  if L4 || L5 || L6
#    ifndef lanai3		/* lanai3-gcc OK for L[456] only */
#      error using wrong compiler
#    endif
#  elif L7 | L8
#    ifndef lanai7		/* lanai7-gcc OK for L[78] only */
#      error using wrong compiler
#    endif
#  else
#    error bad compiler for this version of LANai
#  endif
#endif

/* Define GM_DMA_GRANULARITY to reflect the hardware capabilities. */

#if L4
#define GM_DMA_GRANULARITY 4
#elif L5 
#define GM_DMA_GRANULARITY 8
#elif L6 | L7 | L8 | L9 | LX
#define GM_DMA_GRANULARITY 1
#else
#error Do not know GM_DMA_GRANULARITY
#endif
#if 1
#define GM_DMA_GRANULARITY_ROUNDUP (GM_DMA_GRANULARITY-1)
#else
/* for L4 to check the code with four words alignement assumptions */
#define GM_DMA_GRANULARITY_ROUNDUP 0
#endif

/* Generic GM_PARITY_INT_BIT */

#if L7
#define GM_PARITY_INT_BIT PARITY_INT_BIT
#elif L8 | L9
#define GM_PARITY_INT_BIT PAR_INT_BIT
#elif LX
#define GM_PARITY_INT_BIT PARITY_INT
#endif

#if LX
#define HOST_SIG_BIT REQ_ACK_0
#define GM_LX_SEND_DMA_POINTER DMA0_POINTER
#define GM_LX_RECV_DMA_POINTER DMA1_POINTER
#define GM_LX_SYNC_DMA_POINTER DMA2_POINTER
#define GM_LX_SEND_DMA_COUNT DMA0_COUNT
#define GM_LX_RECV_DMA_COUNT DMA1_COUNT
#define GM_LX_SYNC_DMA_COUNT DMA2_COUNT
#else
#define GM_LX_SEND_DMA_POINTER (*(gm_lp_n_t *) &gm.trash)
#define GM_LX_RECV_DMA_POINTER (*(gm_lp_n_t *) &gm.trash)
#define GM_LX_SYNC_DMA_POINTER (*(gm_lp_n_t *) &gm.trash)
#define GM_LX_SEND_DMA_COUNT (*(gm_u8_n_t *) &gm.trash)
#define GM_LX_RECV_DMA_COUNT (*(gm_u8_n_t *) &gm.trash)
#define GM_LX_SYNC_DMA_COUNT (*(gm_u8_n_t *) &gm.trash)
#endif

#define GM_MSI_AVAILABLE LX

#endif /* _gm_lanai_def_h_ */
