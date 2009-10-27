/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2002 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_lx_recv_h_
#define _gm_lx_recv_h_

#include "gm_simple_types.h"

/* 1.6KB or larger buffer  for 16B blocks */

#define GM_RECEIVE_BUFFER_BYTES (64 * 1024)
#define GM_RECEIVE_BYTES_PER_BLOCK 64
#define GM_BLOCKS_PER_RECEIVE_BUFFER (GM_RECEIVE_BUFFER_BYTES		\
				      / GM_RECEIVE_BYTES_PER_BLOCK)

/****************
 * Received packet descriptor (see LANai X hardware docs)
 ****************/

GM_TYPEDEF_LP_N_T (struct, gm_lx_received_packet_descriptor);

struct gm_lx_received_packet_descriptor
{
  gm_lp_n_t desc_pointer;
  gm_u32_n_t desc_length_with_flags;
};

#define GM_DESC_INVALID (1<<31)
#define GM_DESC_LINK (1<<30)
#define GM_DESC_MARKED_BAD (1<<29)
#define GM_DESC_LAST (1<<28)
#define GM_DESC_ZEROS_OFFSET 25
#define GM_DESC_ZEROS_UNSHIFTED_MASK 0x7
#define GM_DESC_LENGTH_MASK 0xffffff

/****************
 * Received-packet list (see LANai X hardware docs)
 ****************/

struct gm_lx_received_packet_list
{
  struct gm_lx_received_packet_descriptor
  received_packet[GM_BLOCKS_PER_RECEIVE_BUFFER];
};

/****************
 * Receive block
 ****************/

struct gm_lx_receive_block 
{
  gm_u8_n_t bytes[GM_RECEIVE_BYTES_PER_BLOCK];
};

/* Receive buffer.  The hardware documentation refers to the
   combination of a set of receive blocks and the associated
   received-packet descriptors as a "receive buffer". */

GM_TYPEDEF_LP_N_T (struct, gm_receive_buffer);
struct gm_lx_receive_buffer
{
  /* LANai X hardware defined part */
  
  struct gm_lx_receive_block block[GM_BLOCKS_PER_RECEIVE_BUFFER];
  /* 8 */
  struct gm_lx_received_packet_list received_packet_list;
  /* 8 */
};

/* A gm_receive_buffer extended with software state */

GM_TYPEDEF_LP_N_T (struct, gm_lx_extended_receive_buffer);

struct gm_lx_extended_receive_buffer
{
  /* Software extensions to the hardware receive buffer.  These are at
     the front to allow efficient offset addressing, even when the
     size of the receive buffer large. */

  gm_lx_extended_receive_buffer_lp_t next; /* all form a ring */
  gm_u32_n_t ref_cnt;		/* nonzero if in use */

  /* hardware defined part */

  struct gm_lx_receive_buffer receive_buffer;
};

#if GM_BUILDING_FIRMWARE
extern void gm_lx_recv_init (void);
#endif

#endif /* _gm_lx_recv_h_ */
