/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2002 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_mpd_types_h_
#define _gm_mpd_types_h_

#include "gm.h"
#include "gm_debug_mpds.h"
#include "gm_simple_types.h"

/****************************************************************
 * GM packet descriptors
 ****************************************************************/

#define GM_MAX_NUM_MYRINET_PACKET_SEGMENTS 3

GM_TYPEDEF_LP_N_T (struct, gm_myrinet_packet_descriptor);
GM_TYPEDEF_LP_N_T (struct, gm_myrinet_packet_descriptor_free_list);
GM_TYPEDEF_LP_N_T (struct, gm_myrinet_packet_descriptor_list);
GM_TYPEDEF_LP_N_T (struct, gm_myrinet_packet_segment);

typedef struct gm_myrinet_packet_descriptor_free_list
{
  gm_myrinet_packet_descriptor_lp_t first_free;
  gm_u32_n_t nonempty_state_bit;
#if GM_DEBUG_MPDS
  /* 8 */
  gm_u32_n_t debug_cnt;
  gm_myrinet_packet_descriptor_free_list_lp_t debug_next;
  /* 8 */
  gm_const_char_lp_t debug_name;
  gm_u32_n_t reserved;
#endif
} gm_myrinet_packet_descriptor_free_list_t;

typedef struct gm_myrinet_packet_descriptor_list
{
  gm_myrinet_packet_descriptor_lp_t first;
  gm_myrinet_packet_descriptor_lp_t last;
  /* 8 */
  gm_u32_n_t nonempty_state_bit;
  gm_u32_n_t reserved_after_nonempty_state_bit;
#if GM_DEBUG_MPDS
  /* 8 */
  gm_u32_n_t debug_cnt;
  gm_myrinet_packet_descriptor_list_lp_t debug_next;
#endif
} gm_myrinet_packet_descriptor_list_t;

typedef struct gm_myrinet_packet_segment
{
  gm_lp_n_t ptr;
  gm_lp_n_t limit;
} gm_myrinet_packet_segment_t;


typedef void
gm_myrinet_packet_descriptor_callback (gm_lp_t context,
				       gm_myrinet_packet_descriptor_lp_t d);

/*
  Empty arguments in macros invoked undefined behavior in C89 but were a
  common extension. They are expressly allowed in C9X so this can be changed
  back in the future.

  GM_TYPEDEF_LP_N_T (,gm_myrinet_packet_descriptor_callback);
*/
#if GM_MCP
typedef gm_myrinet_packet_descriptor_callback* gm_myrinet_packet_descriptor_callback_lp_t;
#else
typedef gm_lp_n_t gm_myrinet_packet_descriptor_callback_lp_t;
#endif

/* Description of a Myrinet packet in LANai SRAM that is about to
   be sent out the packet interface, or that just arrived via the
   packet interface.

   For receive packets, route and route_limit ignored. */

typedef struct gm_myrinet_packet_descriptor
{
  gm_myrinet_packet_descriptor_free_list_lp_t free_list;
  gm_myrinet_packet_descriptor_lp_t next;
  /* 8 */
  gm_myrinet_packet_descriptor_callback_lp_t free_callback;
  gm_lp_n_t free_callback_context;
  /* 8 */
  gm_myrinet_packet_segment_t route;
  /* 8 */
  gm_myrinet_packet_segment_t header;
  /* 8 */
  gm_myrinet_packet_segment_t payload;
  /* 8 */
  gm_myrinet_packet_segment_t zero;
  /* 8 */
} gm_myrinet_packet_descriptor_t;

#endif /* _gm_mpd_types_h_ */
