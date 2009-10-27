/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_types_h_
#define _gm_types_h_

/* This file defines types used both by the LANai firmware and
   driver code. */

/************************************************************************
 *
 *   ######  #######    #    ######          #     # #######   ###
 *   #     # #         # #   #     #         ##   ## #         ###
 *   #     # #        #   #  #     #         # # # # #         ###
 *   ######  #####   #     # #     #         #  #  # #####      #
 *   #   #   #       ####### #     #         #     # #
 *   #    #  #       #     # #     #         #     # #         ###
 *   #     # ####### #     # ######          #     # #######   ###
 *
 * Because the types declared in this file are shared by the LANai and
 * the host, the structure definitions in this file must satisfy
 * the following requirements to ensure that the structures are
 * formatted identically by the LANai and host compilers:
 *
 * o All types stored in network byte order have type gm_*_n_t.  Since this
 *   file declares all host-accessible types that are stored in network byte
 *   order, this allows for compile-time endian conversion checks.
 * o unsigned integers are declared as gm_uSIZE_n_t, where SIZE is the number
 *   of bits in the integer.  Signed integers are declared as gm_sSIZE_n_t.
 * o C pointer declarations are avoided to make structure sizes independent
 *   of the pointer size of the compiler being used:
 *   - All LANai addresses have types ending in "_lp_n_t".  These types are
 *     defined as C pointers on the LANai and as gm_u32_n_t on the host.
 *   - All user virtual address have type "gm_up*_t".  These types
 *     are defined as C pointers on the host and as gm_u*_n_t in the lanai.
 *   - All DMA addresses have type "gm_dp*_t".
 * o All structure fields are aligned (using explicit padding if needed)
 *   to prevent different compilers from packing structures differently.
 * o Bitfields and enumerated types are avoided for the same reason.
 * o No zero-length arrays.
 *
 * Other conventions used in this file:
 * o All global-scope identifiers start with "_*[Gg][Mm]_" (regexp notation)
 * o All typedefs end in "_n_t"
 * o The Gnu coding standards are followed.
 ************************************************************************/

/***********************************************************************
 * Configuration defaults:
 ***********************************************************************/

#include "gm.h"
#include "gm_bit_twiddling.h"
#include "gm_bitmap.h"
#include "gm_config.h"
#include "gm_cpp.h"
#include "gm_debug_counters.h"
#include "gm_debug_lanai_dma_types.h"
#include "gm_dma_perf.h"
#include "gm_enable_error_counters.h"
#include "gm_enable_fast_small_send.h"
#include "gm_enable_log_dispatch.h"
#include "gm_enable_trace.h"
#include "gm_error_counters.h"
#include "gm_ether.h"
#include "gm_gateway_cache.h"
#include "gm_lx_recv.h"
#include "gm_mapper_state.h"
#include "gm_mpd_types.h"
#include "gm_simple_types.h"
#include "gm_sizes.h"
#include "gm_trace.h"
#include "gm_yp.h"

/* BAD: Only constants that are constant across all GM implementations
   should be here. */

/* The size of pages in host memory.  For machines without VM, it is
   convenient to pretend there are 4K pages, since the
   gm_port_protected_lanai_side structures are made to have length
   GM_PAGE_LEN. */
#if GM_ENABLE_VM
#  if !GM_PAGE_LEN && GM_MCP
#    error Must define GM_PAGE_LEN if GM_ENABLE_VM is set.
#  endif
#endif

/****************
 * Sizeof declarations
 ****************/

/* The declarations here must agree with the data types, and should be
   declared here only if compiler sizeof() directive is not
   sufficient for our purposes.  Any declaration here should be
   checked after the struct declaration with GM_CHECK_SIZEOF(). */

/* Used to define GM_NUM_RECV_TOKENS */
#define GM_SIZEOF_HOST_RECV_TOKEN		(GM_SIZEOF_UP_T == 8 ? 16 : 8)
/* Used to define GM_NUM_SEND_TOKENS */
#define GM_SIZEOF_SEND_QUEUE_SLOT		32

/* A macro used to verify the preceeding declarations after the actual
   data type definition.  It will generate a compile-time error. */

/* The number of user ports supported.  Must be <= 16 in the current
   GM version due to limitations of the connection's
   active_subport_bitmask field. */
#define GM_NUM_PORTS_STANDARD			16
#define GM_NUM_PORTS_LOWER                      8

#if GM_SUPPORT_64K_PAGES
/* using 64k pages takes up a lot of memory in the lanai, 
   so we need to lower the number of ports */
#define GM_NUM_PORTS                            GM_NUM_PORTS_LOWER
#else
#define GM_NUM_PORTS				GM_NUM_PORTS_STANDARD
#endif

/* Here we make the SRAM window length be 1 page, since the kernel can
   only map full pages and we don't want to waste any memory to pad
   the SRAM window.  However, we limit the SRAM window length to 8K to
   ensure that the MCP will fit in NICs with 1MB of SRAM.

   You can change GM_PORT_SRAM_WINDOW_LEN to a larger power of two to
   get more send and receive tokens for each GM port, but making it
   too large can make the firmware too large to load on your hardware,
   and increasing it decreases the memory available for page table
   caching and reduces the maximum size of Myrinet network supported
   by the firmware. */

#define GM_PORT_HOST_RECV_TOKEN_QUEUE_LEN (GM_NUM_SEND_QUEUE_SLOTS * GM_SIZEOF_SEND_QUEUE_SLOT)
#define GM_PORT_HOST_SEND_TOKEN_QUEUE_LEN ((GM_NUM_RECV_TOKENS + 2) * GM_SIZEOF_HOST_RECV_TOKEN)
#define GM_PORT_SRAM_WINDOW_LEN (GM_PORT_HOST_RECV_TOKEN_QUEUE_LEN + GM_PORT_HOST_SEND_TOKEN_QUEUE_LEN)

#define GM_PORT_MAPPED_SRAM_LEN ((GM_PORT_SRAM_WINDOW_LEN > GM_PAGE_LEN) \
				 ? GM_SLOW_POW2_ROUNDUP(GM_PORT_SRAM_WINDOW_LEN) : GM_PAGE_LEN)

				/* BAD: Can (and therefore should)
				   make these bigger on machines with
				   pages larger than 4K. */

/* Number of send tokens. */
#define GM_NUM_SEND_TOKENS (GM_NUM_SEND_QUEUE_SLOTS - 3)

/* Number of recv tokens.  2^n - 2, where 2^n of them are guaranteed
   to fit in half a page. */


#define GM_NUM_RECV_TOKENS 254

/* # of ethernet recv tokens must fit in a _gm_recv_event and a
   gm_host_recv_token tag, currently 8 bits */
   
#define GM_NUM_ETHERNET_RECV_TOKENS 		((GM_NUM_RECV_TOKENS > 255) \
                                                 ? 255 			    \
						 : GM_NUM_RECV_TOKENS)

/* The number of bins to be used in the hash table for storing receive
   tokens. 2^N */
#define GM_RECV_TOKEN_HASH_BINS			64

/* Twice the number of page hash table entries to be cached in the
   LANai, minus 1. 2^N-1 */

#define GM_MAX_PAGE_HASH_CACHE_INDEX (gm.page_hash.cache.max_index)

/* The maximum number of bytes for a streamlined receive (one that
   requires a host-side copy, but one less DMA). 2^N, and a multiple
   GM_RDMA_GRANULARITY. */
#define GM_MAX_FAST_RECV_BYTES			128

/* MTU to be told to the IP layer above you is GM_IP_MTU */
#define GM_IP_MTU       (9000)

/* The number of GET reply buffers on LX */
#define GM_NUM_GET_REPLY_BUFFERS 8

/****
 * Debugging options
 ****/
#define GM_LOG_LEN		(GM_LOG_DISPATCHES ? 150 : 1)
#define GM_DISPATCH_MAX_NUM	(GM_LOG_DISPATCHES ? 300 : 1)

/***********************************************************************
 * Macros
 ***********************************************************************/

/* Misc. nonconfigurable constants. */

#define GM_NULL 		0
#define GM_TRUE 		1
#define GM_FALSE 		0
#define GM_SIZEOF_PACKET_HEADER	24
#define GM_MTU			4096
#define GM_MIN_MESSAGE_SIZE	3
/*
#define GM_MAX_DMA_CTR 		((GM_ETHERNET_MTU			\
				  > GM_MTU+2*GM_MAX_DMA_GRANULARITY)	\
				 ? GM_ETHERNET_MTU			\
				 :GM_MTU+2*GM_MAX_DMA_GRANULARITY)
*/
#define __GM_MAX(A, B) ((A) > (B) ? (A) : (B))
#define GM_MAX_DMA_CTR		__GM_MAX (GM_PAGE_LEN, GM_MYRINET_MTU)
#define GM_MYRINET_MTU							\
(__GM_MAX (GM_ETHERNET_OVER_MYRINET_MTU,				\
	   GM_SIZEOF_PACKET_HEADER + GM_MTU)				\
 + 8)
#define GM_DAEMON_PORT_ID	0
#define GM_MAPPER_PORT_ID	1
#define GM_ETHERNET_PORT_ID	3

#define GM_CRC_TYPE		char
#define GM_CRC32_TYPE		unsigned int

#define GM_NO_ROUTE		((gm_u8_t) 0xFF)

				/* Implicit configuration definitions */
#define GM_NUM_HOST_PAGES	(gm_total_host_mem/GM_HOST_PAGE_LENGTH)
#define GM_NUM_SUBPORTS		(GM_NUM_PORTS * GM_NUM_SEND_TOKENS)

/* Number of USED slots in a queue.  An extra slot might be allocated
   to streamline wraparound. */
				/* 1 extra to set alarm, and *2* extra
				   to flush alarm */
#define GM_NUM_SEND_QUEUE_SLOTS 64
/* extra for GM_SLEEP and GM_SENT*_EVENTs */
#define GM_NUM_RECV_QUEUE_SLOTS (GM_NUM_RECV_TOKENS+GM_NUM_SEND_QUEUE_SLOTS+1)
#define GM_NUM_RECV_TOKEN_QUEUE_SLOTS (GM_NUM_RECV_TOKENS)
#define GM_RECV_QUEUE_SLOTS_PER_PAGE					\
(GM_PAGE_LEN / sizeof (gm_recv_queue_slot_t))
/* Must be more than GM_NUM_SUBPORTS, even, and the smaller the better. */
#define GM_NUM_SEND_RECORDS 	(GM_NUM_SUBPORTS + 2)

/* The maximum number of DMA segments used by the firmware in normal
   operation. */

#define _GM_MAX_NUM_DMA_SEGMENTS					\
(2 /* paranoia HACK */							\
 * 2 /* number of DMA engines */					\
 * __GM_MAX (GM_MAX_ETHERNET_SCATTER_CNT, 2) /* # queued by each */	\
 * 3 /* Thrice as many to allow GM_SPARC_STREAMING DMA cache		\
	flushing */							\
 + 1 /* Because one GM_SPARC_STREAMING descriptor may be in progress	\
	after the wake */						\
 + 1 /* chained token DMA */						\
 )

/* The maximum number of DMA segments used by the firmware for DMA
   performance DMA performance testing. */

GM_TOP_LEVEL_ASSERT (GM_MAX_PAGE_LEN % GM_MTU == 0);

/* Account for extra DMA segments used by sparc streaming */
#define _GM_MAX_NUM_DMA_TEST_SEGMENTS					\
(2 * (GM_NUM_DMA_TEST_SEGMENTS + 2)					\
 * (GM_ENABLE_SPARC_STREAMING && (GM_CPU_sparc || GM_CPU_sparc64) ? 2 : 1))

#define GM_MAX_NUM_DMA_SEGMENTS						\
__GM_MAX (_GM_MAX_NUM_DMA_TEST_SEGMENTS, _GM_MAX_NUM_DMA_SEGMENTS)

/************************************************************************
 * Utility macros
 ************************************************************************/

/* Declare a pad to come after an x byte object to pad up to 2^n bytes.
   For example,
   
   sizeof (struct foo { struct bar; GM_POW2_PAD(sizeof(struct bar)); })
   
   would be a power of 2. */
   
#define GM_POW2_PAD(x) gm_u8_n_t pad[GM_SLOW_POW2_ROUNDUP (x) - (x)]

/* Internal host pointer macros */
#if !GM_MCP
#  if GM_SIZEOF_VOID_P == 8
#    define _GM_ptr_ALIGN(n,m) ((void *)((gm_u64_t)(n)&~((gm_u64_t)(m)-1)))
#    define _GM_ptr_ROUNDUP(n,m) ((void *)(((gm_u64_t)(n)+(gm_u64_t)(m)-1)&~((gm_u64_t)(m)-1)))
#  elif GM_SIZEOF_VOID_P == 4
#    define _GM_ptr_ALIGN(n,m) ((void *)((gm_u32_t)(n)&~((gm_u32_t)(m)-1)))
#    define _GM_ptr_ROUNDUP(n,m) ((void *)(((gm_u32_t)(n)+(gm_u32_t)(m)-1)&~((gm_u32_t)(m)-1)))
#  endif
#  define _GM_ptr_ROUND_DOWN(n,m) _GM_ptr_ALIGN(n,m)
#endif
#if GM_SIZEOF_UP_T == 8
#  define _GM_up_ALIGN(n,m) ((gm_up_t)((gm_u64_t)(n)&~((gm_u64_t)(m)-1)))
#  define _GM_up_ROUNDUP(n,m) ((gm_up_t)(((gm_u64_t)(n)+(gm_u64_t)(m)-1)&~((gm_u64_t)(m)-1)))
#elif GM_SIZEOF_UP_T == 4
#  define _GM_up_ALIGN(n,m) ((gm_up_t)((gm_u32_t)(n)&~((gm_u32_t)(m)-1)))
#  define _GM_up_ROUNDUP(n,m) ((gm_up_t)(((gm_u32_t)(n)+(gm_u32_t)(m)-1)&~((gm_u32_t)(m)-1)))
#endif
#define _GM_up_ROUND_DOWN(n,m) _GM_up_ALIGN(n,m)
/* Internal lanai pointer macros */
#define _GM_lp_ALIGN(n,m) ((gm_lp_t)((gm_u32_t)(n)&~((gm_u32_t)(m)-1)))
#define _GM_lp_ROUNDUP(n,m) ((gm_lp_t)(((gm_u32_t)(n)+(gm_u32_t)(m)-1)&~((gm_u32_t)(m)-1)))
#define _GM_lp_ROUND_DOWN(n,m) _GM_lp_ALIGN(n,m)
/* Internal DMA pointer macros */
#define _GM_dp_ALIGN(n,m) ((gm_dp_t)((gm_dp_t)(n)&~((gm_dp_t)(m)-1)))
#define _GM_dp_ROUNDUP(n,m) ((gm_dp_t)(((gm_dp_t)(n)+(gm_dp_t)(m)-1)&~((gm_dp_t)(m)-1)))
#define _GM_dp_ROUND_DOWN(n,m) _GM_dp_ALIGN(n,m)
/* Internal u32 macros */
#define _GM_u32_ALIGN(n,m) ((gm_u32_t)(n)&~((gm_u32_t)(m)-1))
#define _GM_u32_ROUNDUP(n,m) (((gm_u32_t)(n)+(gm_u32_t)(m)-1)&~((gm_u32_t)(m)-1))
#define _GM_u32_ROUND_DOWN(n,m) _GM_u32_ALIGN(n,m)
/* Internal u64 macros */
#define _GM_u64_ALIGN(n,m) ((gm_u64_t)(n)&~((gm_u64_t)(m)-1))
#define _GM_u64_ROUNDUP(n,m) (((gm_u64_t)(n)+(gm_u64_t)(m)-1)&~((gm_u64_t)(m)-1))
#define _GM_u64_ROUND_DOWN(n,m) _GM_u64_ALIGN(n,m)
/* Internal size macros */
#define _GM_size_ALIGN(n,m) ((gm_size_t)(n)&~((gm_size_t)(m)-1))
#define _GM_size_ROUNDUP(n,m) (((gm_size_t)(n)+(gm_size_t)(m)-1)&~((gm_size_t)(m)-1))
#define _GM_size_ROUND_DOWN(n,m) _GM_size_ALIGN(n,m)
/****
 * Generic alignment macros
 ****/
#define _GM_ALIGN(type,n,m) type##_ALIGN(n,m)
#define _GM_ROUNDUP(type,n,m) type##_ROUNDUP(n,m)
#define _GM_ROUND_DOWN(type,n,m) type##_ROUND_DOWN(n,m)

#define GM_ALIGN(type,n,m) _GM_##type##_ALIGN(n,m)
#define GM_ROUNDUP(type,n,m) _GM_##type##_ROUNDUP(n,m)
#define GM_ROUND_DOWN(type,n,m) _GM_##type##_ROUND_DOWN(n,m)

#define GM_NATURALLY_ALIGNED(ptr) (GM_POWER_OF_TWO (sizeof (*ptr))	\
				   && GM_ALIGNED (ptr, sizeof (*ptr)))
/* GM_ALIGNED is in gm.h */
/****
 * Alignment macros
 ****/
/* BAD: all GM alignment requirements will be lifted */
#define GM_PACKET_GRANULARITY   8
#define GM_DMA_ROUNDUP(t,p) 	_GM_ROUNDUP (_GM_##t,p, GM_DMA_GRANULARITY)
#define GM_DMA_ALIGN(t,p) 	_GM_ALIGN (_GM_##t,p, GM_DMA_GRANULARITY)
#if GM_MCP
#define GM_RDMA_ROUNDUP(t,p) 	_GM_ROUNDUP (_GM_##t,p, GM_RDMA_GRANULARITY)
#define GM_RDMA_ALIGN(t,p) 	_GM_ALIGN (_GM_##t,p, GM_RDMA_GRANULARITY)
#define GM_RDMA_ALIGNED(p)	GM_ALIGNED (p, GM_RDMA_GRANULARITY)
#define GM_PACKET_ROUNDUP(t,p) 	_GM_ROUNDUP (_GM_##t,p, GM_PACKET_GRANULARITY)
#define GM_PACKET_ALIGN(t,p) 	_GM_ALIGN (_GM_##t,p, GM_PACKET_GRANULARITY)
#define GM_PACKET_ALIGNED(p)	GM_ALIGNED (p, GM_PACKET_GRANULARITY)
#endif
/* Page macros */
#define GM_PAGE_ROUNDUP(t,p) 	_GM_ROUNDUP (_GM_##t,p, GM_PAGE_LEN)
#define GM_PAGE_ALIGN(t,p) 	_GM_ALIGN (_GM_##t,p, GM_PAGE_LEN)
#define GM_PAGE_ALIGNED(p)	GM_ALIGNED (p, GM_PAGE_LEN)
#define GM_PAGE_OFFSET(n)	((gm_u32_t)((gm_size_t)(n)) & (GM_PAGE_LEN-1))
#define GM_PAGE_REMAINING(n)	(GM_PAGE_LEN - GM_PAGE_OFFSET (n))
#define GM_PAGE_ROUND_DOWN(t,p) _GM_ALIGN(_GM_##t,p,GM_PAGE_LEN)
#define GM_PAGE_SHIFT (GM_PAGE_LEN == 16384 ? 14 :			\
		       GM_PAGE_LEN == 8192 ? 13 :			\
		       12)
#define GM_DMA_PAGE_ADDR(n)	((gm_dp_t)(n) << GM_PAGE_SHIFT)
#define GM_UP_PAGE_NUM(a) 	((gm_up_t)(a) >> GM_PAGE_SHIFT)
#define GM_DP_PAGE_NUM(a) 	((gm_dp_t)(a) >> GM_PAGE_SHIFT)
/* Ethernet addres table macros */
/* enough to support ethernet addresses for 64K nodes & 4K pages:
   64K * 8B / 4KB */
#define GM_MAX_NUM_ADDR_TABLE_PIECES ((1<<16) * sizeof (gm_unique_id_64_t) \
				      / 4096)
/* Page hash table macros */
#define GM_ENTRIES_PER_PIECE (GM_PAGE_LEN/sizeof(gm_pte_t))
#if !GM_ENABLE_VM
#define GM_PAGE_HASH_PIECE_REF_TABLE_LEN 0
#define GM_MAX_NUM_HOST_HASH_TABLE_PIECES 0
#define GM_PAGE_HASH_MAX_INDEX 0
#define GM_PAGE_HASH_MAX_INDEX 0
#else /* GM_ENABLE_VM */
/* enough pages for 4GB of registered memory */
#define GM_PAGE_HASH_PIECE_REF_TABLE_LEN				\
(GM_PAGE_LEN == 4096 ? GM_PAGE_LEN * 16 :				\
 GM_PAGE_LEN == 8192 ? GM_PAGE_LEN * 2 :				\
 GM_PAGE_LEN)
#define GM_MAX_NUM_HOST_HASH_TABLE_PIECES 				\
(GM_PAGE_HASH_PIECE_REF_TABLE_LEN / sizeof (gm_dp_n_t))
#define GM_PAGE_HASH_MAX_INDEX ((GM_MAX_NUM_HOST_HASH_TABLE_PIECES	\
				 * GM_ENTRIES_PER_PIECE)		\
	                        - 1)
#endif /* GM_ENABLE_VM */
  /* This should never be larger than
     32bit, so OK to use
     GM_DMA_PAGE_ROUNDUP */
#define GM_RECV_QUEUE_ALLOC_LEN \
     GM_PAGE_ROUNDUP (u32, (GM_NUM_RECV_QUEUE_SLOTS \
				* sizeof (gm_recv_queue_slot_t)))
  /* Macros to combine/extract port id
     and priority into/outof a subport
     ID. */
#define GM_SUBPORT(priority,port) ((port)<<1|priority)
#define GM_SUBPORT_PRIORITY(subport_id) ((subport_id)&1)
#define GM_SUBPORT_PORT(subport_id) ((unsigned int) (subport_id)>>1U)
#define GM_MAX_SUBPORT_ID GM_SUBPORT (GM_HIGH_PRIORITY, GM_NUM_PORTS - 1)
/***********************************************************************
 * enumerations
 ***********************************************************************/
/* Aliases for priorities */
#define GM_MIN_PRIORITY	   GM_LOW_PRIORITY
#define GM_MAX_PRIORITY	   GM_HIGH_PRIORITY
/* Packet magic numbers.  These are actually the Myrinet packet
   "types" required in the first 2 bytes of every packet.  However, we
   refer to them as the packet magic number in GM source code to avoid
   confusion with the GM packet "type". */
enum gm_packet_type
{
  /* Contact help@myri.com to have one allocated for your
     project. See http://www.myri.com/scs/types.html */
  GM_PACKET_TYPE = 0x0008,
  GM_ETHERNET_PACKET_TYPE = 0x0009,
  GM_MAPPING_PACKET_TYPE = 0x700f
};

/* Node types that may be on the fabric */
enum gm_node_type
{
  GM_NODE_TYPE_GM = 0,		/* GM node */
  GM_NODE_TYPE_XM = 1,		/* XM node */
  GM_NODE_TYPE_MX = 2,		/* MX node */
  GM_NODE_TYPE_UNKNOWN = 98,	/* unknown node type */
  GM_NODE_TYPE_PURGED = 99	/* used to track "purged" nodes */
};

/****************
 * GM packet subtypes
 ****************/

/* GM before GM-2.0 used packet types 0-133.  GM-2.0 subtypes start at
   256 to ensure that pre-GM-2.0 packets are ignored. */

#define GM_SUBTYPE(n) (256 + (n))

/* The types of GM packets.

   NOTE: although this is an enum, we explicitly indicate the values
   to prevent accidentally changing them, since they should NEVER be
   changed, to ensure backwards compatibility. */

enum gm_packet_subtype
{
  GM_NULL_SUBTYPE = GM_SUBTYPE (0),
  GM_BOOT_SUBTYPE = GM_SUBTYPE (1),
  GM_GET_SUBTYPE = GM_SUBTYPE (2),
  GM_GOT_SUBTYPE = GM_SUBTYPE (3),
  /* 4..6 */
  /*  GM_RAW_HACK_SUBTYPE = GM_SUBTYPE (6), */
  /* The following types with sizes */
  /* allow for more compact header */
  /* encoding. */

  /* Identifiers for unsegmented messages. */
  GM_RELIABLE_DATA_SUBTYPE_0 = GM_SUBTYPE (7),
  GM_RELIABLE_DATA_SUBTYPE_1 = GM_SUBTYPE (8),
  GM_RELIABLE_DATA_SUBTYPE_2 = GM_SUBTYPE (9),
  GM_RELIABLE_DATA_SUBTYPE_3 = GM_SUBTYPE (10),
  GM_RELIABLE_DATA_SUBTYPE_4 = GM_SUBTYPE (11),
  GM_RELIABLE_DATA_SUBTYPE_5 = GM_SUBTYPE (12),
  GM_RELIABLE_DATA_SUBTYPE_6 = GM_SUBTYPE (13),
  GM_RELIABLE_DATA_SUBTYPE_7 = GM_SUBTYPE (14),
  GM_RELIABLE_DATA_SUBTYPE_8 = GM_SUBTYPE (15),
  GM_RELIABLE_DATA_SUBTYPE_9 = GM_SUBTYPE (16),
  GM_RELIABLE_DATA_SUBTYPE_10 = GM_SUBTYPE (17),
  GM_RELIABLE_DATA_SUBTYPE_11 = GM_SUBTYPE (18),
  GM_RELIABLE_DATA_SUBTYPE_12 = GM_SUBTYPE (19),
  GM_RELIABLE_DATA_SUBTYPE_13 = GM_SUBTYPE (20),
  GM_RELIABLE_DATA_SUBTYPE_14 = GM_SUBTYPE (21),
  GM_RELIABLE_DATA_SUBTYPE_15 = GM_SUBTYPE (22),
  GM_RELIABLE_DATA_SUBTYPE_16 = GM_SUBTYPE (23),
  GM_RELIABLE_DATA_SUBTYPE_17 = GM_SUBTYPE (24),
  GM_RELIABLE_DATA_SUBTYPE_18 = GM_SUBTYPE (25),
  GM_RELIABLE_DATA_SUBTYPE_19 = GM_SUBTYPE (26),
  GM_RELIABLE_DATA_SUBTYPE_20 = GM_SUBTYPE (27),
  GM_RELIABLE_DATA_SUBTYPE_21 = GM_SUBTYPE (28),
  GM_RELIABLE_DATA_SUBTYPE_22 = GM_SUBTYPE (29),
  GM_RELIABLE_DATA_SUBTYPE_23 = GM_SUBTYPE (30),
  GM_RELIABLE_DATA_SUBTYPE_24 = GM_SUBTYPE (31),
  GM_RELIABLE_DATA_SUBTYPE_25 = GM_SUBTYPE (32),
  GM_RELIABLE_DATA_SUBTYPE_26 = GM_SUBTYPE (33),
  GM_RELIABLE_DATA_SUBTYPE_27 = GM_SUBTYPE (34),
  GM_RELIABLE_DATA_SUBTYPE_28 = GM_SUBTYPE (35),
  GM_RELIABLE_DATA_SUBTYPE_29 = GM_SUBTYPE (36),
  GM_RELIABLE_DATA_SUBTYPE_30 = GM_SUBTYPE (37),
  GM_RELIABLE_DATA_SUBTYPE_31 = GM_SUBTYPE (38),

  /* Identifiers for the first segment
     of a segmented message. */

  GM_RELIABLE_HEAD_SUBTYPE_0 = GM_SUBTYPE (39),
  GM_RELIABLE_HEAD_SUBTYPE_1 = GM_SUBTYPE (40),
  GM_RELIABLE_HEAD_SUBTYPE_2 = GM_SUBTYPE (41),
  GM_RELIABLE_HEAD_SUBTYPE_3 = GM_SUBTYPE (42),
  GM_RELIABLE_HEAD_SUBTYPE_4 = GM_SUBTYPE (43),
  GM_RELIABLE_HEAD_SUBTYPE_5 = GM_SUBTYPE (44),
  GM_RELIABLE_HEAD_SUBTYPE_6 = GM_SUBTYPE (45),
  GM_RELIABLE_HEAD_SUBTYPE_7 = GM_SUBTYPE (46),
  GM_RELIABLE_HEAD_SUBTYPE_8 = GM_SUBTYPE (47),
  GM_RELIABLE_HEAD_SUBTYPE_9 = GM_SUBTYPE (48),
  GM_RELIABLE_HEAD_SUBTYPE_10 = GM_SUBTYPE (49),
  GM_RELIABLE_HEAD_SUBTYPE_11 = GM_SUBTYPE (50),
  GM_RELIABLE_HEAD_SUBTYPE_12 = GM_SUBTYPE (51),
  GM_RELIABLE_HEAD_SUBTYPE_13 = GM_SUBTYPE (52),
  GM_RELIABLE_HEAD_SUBTYPE_14 = GM_SUBTYPE (53),
  GM_RELIABLE_HEAD_SUBTYPE_15 = GM_SUBTYPE (54),
  GM_RELIABLE_HEAD_SUBTYPE_16 = GM_SUBTYPE (55),
  GM_RELIABLE_HEAD_SUBTYPE_17 = GM_SUBTYPE (56),
  GM_RELIABLE_HEAD_SUBTYPE_18 = GM_SUBTYPE (57),
  GM_RELIABLE_HEAD_SUBTYPE_19 = GM_SUBTYPE (58),
  GM_RELIABLE_HEAD_SUBTYPE_20 = GM_SUBTYPE (59),
  GM_RELIABLE_HEAD_SUBTYPE_21 = GM_SUBTYPE (60),
  GM_RELIABLE_HEAD_SUBTYPE_22 = GM_SUBTYPE (61),
  GM_RELIABLE_HEAD_SUBTYPE_23 = GM_SUBTYPE (62),
  GM_RELIABLE_HEAD_SUBTYPE_24 = GM_SUBTYPE (63),
  GM_RELIABLE_HEAD_SUBTYPE_25 = GM_SUBTYPE (64),
  GM_RELIABLE_HEAD_SUBTYPE_26 = GM_SUBTYPE (65),
  GM_RELIABLE_HEAD_SUBTYPE_27 = GM_SUBTYPE (66),
  GM_RELIABLE_HEAD_SUBTYPE_28 = GM_SUBTYPE (67),
  GM_RELIABLE_HEAD_SUBTYPE_29 = GM_SUBTYPE (68),
  GM_RELIABLE_HEAD_SUBTYPE_30 = GM_SUBTYPE (69),
  GM_RELIABLE_HEAD_SUBTYPE_31 = GM_SUBTYPE (70),

  /* Identifiers for continuations of
     segmented messages. */
  GM_RELIABLE_BODY_SUBTYPE = GM_SUBTYPE (71),
  GM_RELIABLE_TAIL_SUBTYPE = GM_SUBTYPE (72),

  GM_PUT_SUBTYPE = GM_SUBTYPE (73),
  GM_PUT_HEAD_SUBTYPE = GM_SUBTYPE (74),
  GM_PUT_BODY_SUBTYPE = GM_SUBTYPE (75),    
  GM_PUT_TAIL_SUBTYPE = GM_SUBTYPE (76),    
					    
  GM_DATAGRAM_SUBTYPE_0 = GM_SUBTYPE (77),  
  GM_DATAGRAM_SUBTYPE_1 = GM_SUBTYPE (78),  
  GM_DATAGRAM_SUBTYPE_2 = GM_SUBTYPE (79),  
  GM_DATAGRAM_SUBTYPE_3 = GM_SUBTYPE (80),  
  GM_DATAGRAM_SUBTYPE_4 = GM_SUBTYPE (81),  
  GM_DATAGRAM_SUBTYPE_5 = GM_SUBTYPE (82),  
  GM_DATAGRAM_SUBTYPE_6 = GM_SUBTYPE (83),  
  GM_DATAGRAM_SUBTYPE_7 = GM_SUBTYPE (84),  
  GM_DATAGRAM_SUBTYPE_8 = GM_SUBTYPE (85),  
  GM_DATAGRAM_SUBTYPE_9 = GM_SUBTYPE (86),  
  GM_DATAGRAM_SUBTYPE_10 = GM_SUBTYPE (87), 
  GM_DATAGRAM_SUBTYPE_11 = GM_SUBTYPE (88), 
  GM_DATAGRAM_SUBTYPE_12 = GM_SUBTYPE (89), 
  GM_DATAGRAM_SUBTYPE_13 = GM_SUBTYPE (90), 
  GM_DATAGRAM_SUBTYPE_14 = GM_SUBTYPE (91), 
  GM_DATAGRAM_SUBTYPE_15 = GM_SUBTYPE (92), 
  GM_DATAGRAM_SUBTYPE_16 = GM_SUBTYPE (93), 
  GM_DATAGRAM_SUBTYPE_17 = GM_SUBTYPE (94), 
  GM_DATAGRAM_SUBTYPE_18 = GM_SUBTYPE (95), 
  GM_DATAGRAM_SUBTYPE_19 = GM_SUBTYPE (96), 
  GM_DATAGRAM_SUBTYPE_20 = GM_SUBTYPE (97), 
  GM_DATAGRAM_SUBTYPE_21 = GM_SUBTYPE (98), 
  GM_DATAGRAM_SUBTYPE_22 = GM_SUBTYPE (99), 
  GM_DATAGRAM_SUBTYPE_23 = GM_SUBTYPE (100),
  GM_DATAGRAM_SUBTYPE_24 = GM_SUBTYPE (101), 
  GM_DATAGRAM_SUBTYPE_25 = GM_SUBTYPE (102),
  GM_DATAGRAM_SUBTYPE_26 = GM_SUBTYPE (103), 
  GM_DATAGRAM_SUBTYPE_27 = GM_SUBTYPE (104),
  GM_DATAGRAM_SUBTYPE_28 = GM_SUBTYPE (105), 
  GM_DATAGRAM_SUBTYPE_29 = GM_SUBTYPE (106),
  GM_DATAGRAM_SUBTYPE_30 = GM_SUBTYPE (107), 
  GM_DATAGRAM_SUBTYPE_31 = GM_SUBTYPE (108),
				       	
  GM_YP_REPLY_SUBTYPE = GM_SUBTYPE (109),
  GM_YP_QUERY_SUBTYPE = GM_SUBTYPE (110),
  
  /****************
   * Control messages
   *
   * These messages need not wait on the DMA engine.
   ****************/

  GM_MIN_CONTROL_PACKET_SUBTYPE = GM_SUBTYPE (128),
  
  GM_ACK_SUBTYPE = GM_SUBTYPE (128),
  GM_NACK_SUBTYPE = GM_SUBTYPE (129),
  GM_NACK_DOWN_SUBTYPE = GM_SUBTYPE (130),
  GM_NACK_REJECT_SUBTYPE = GM_SUBTYPE (131),
  GM_NACK_OPEN_CONNECTION_SUBTYPE = GM_SUBTYPE (132),
  GM_NACK_CLOSE_CONNECTION_SUBTYPE = GM_SUBTYPE (133),
  
  GM_MAX_CONTROL_PACKET_SUBTYPE = GM_NACK_CLOSE_CONNECTION_SUBTYPE
};					      

enum gm_send_token_type
{
  GM_ST_RELIABLE = 0,
  GM_ST_PUT = 1,
  GM_ST_RAW = 2,
  GM_ST_PROBE = 3,
  GM_ST_GET = 4,
  /* 5 */
  GM_ST_ETHERNET_SEND = 6,
  /* 7 */
  GM_ST_DATAGRAM = 8,
  GM_ST_PIO_DATAGRAM = 9
};

enum gm_interrupt_type
{
  GM_NO_INTERRUPT,
  GM_INTERRUPTS_UNINITIALIZED,
  GM_PAUSE_INTERRUPT,
  GM_COMMAND_COMPLETE_INTERRUPT,
  GM_WAKE_INTERRUPT,
  GM_PRINT_INTERRUPT,
  GM_FAILED_ASSERTION_INTERRUPT,
  GM_WRITE_INTERRUPT,
  GM_WAIT_INTERRUPT,
  GM_BOGUS_RECV_INTERRUPT,
  GM_BOGUS_SEND_INTERRUPT,
  GM_YP_WAKE_INTERRUPT,
  GM_NODE_ID_TRANSLATION_INTERRUPT
};

/***********************************************************************
 * Types
 ***********************************************************************/

/* handy type for implying that an int acts as a boolean. */
typedef int gm_boolean_t;

/************
 * LANai-side pointers
 ************/

/* *INDENT-OFF* */
GM_LP_N_T (gm_u8_t) gm_u8_lp_n_t;
GM_LP_N_T (gm_s8_t) gm_s8_lp_n_t;
GM_LP_N_T (void *) gm_lp_n_lp_n_t;
GM_LP_N_T (gm_up_n_t) gm_up_n_lp_n_t;
/* *INDENT-ON* */

/* *INDENT-OFF* */
GM_TYPEDEF_LP_N_T (struct, _gm_sent_token_report);
GM_TYPEDEF_LP_N_T (struct, gm_connection);
GM_TYPEDEF_LP_N_T (struct, gm_dma_descriptor);
GM_TYPEDEF_LP_N_T (struct, gm_ethernet_recv_token);
GM_TYPEDEF_LP_N_T (struct, gm_ethernet_send_event);
GM_TYPEDEF_LP_N_T (struct, gm_firmware_log_entry);
GM_TYPEDEF_LP_N_T (struct, gm_host_recv_token);
GM_TYPEDEF_LP_N_T (struct, gm_port_protected_lanai_side);
GM_TYPEDEF_LP_N_T (struct, gm_port_unprotected_lanai_side);
GM_TYPEDEF_LP_N_T (struct, gm_recv_token);
GM_TYPEDEF_LP_N_T (struct, gm_send_queue_slot);
GM_TYPEDEF_LP_N_T (struct, gm_send_record);
GM_TYPEDEF_LP_N_T (struct, gm_subport);
GM_TYPEDEF_LP_N_T (struct, gm_yp_query_context);
GM_TYPEDEF_LP_N_T (union, gm_cached_pte);
GM_TYPEDEF_LP_N_T (union, gm_send_token);

/* *INDENT-ON* */

#if GM_MCP
typedef void (*gm_handler_n_t)(void);
#else
typedef gm_lp_n_t gm_handler_n_t;
#endif

/* A GM "sexno" is a structure containing both the session number
   ("sesno") and sequence number within the session ("seqno").  While
   most communication uses only sequence numbers, GM uses the session
   numbers to reliably multiplex many steams of data over a single
   connection, avoiding the connection state overhead that would be
   required if multiple connections were used. */
typedef union gm_sexno
{
  gm_s32_n_t whole;
  struct
  {
    gm_s16_n_t sesno;		/* session number */
    gm_s16_n_t seqno;		/* sequence number */
  }
  parts;
}

gm_sexno_t;

/****
 * Page hash table types
 ****/

/* A page is just GM_PAGE_LEN bytes.  This structure is useful for
   computing page numbers without casting to an integer type. */
#ifdef gm_lanai
#error
typedef struct gm_page
{
  gm_s8_n_t byte[GM_PAGE_LEN];
}

gm_page_t;
#endif

/* An page hash table entry used to map (virt_page,port)->dma_page.
   That is, the virt_page and port fields act as a key, and the
   dma_page is the value. 2^N bytes.

   WARNING: All fields are stored in network byte order.. */

typedef struct gm_pte
{
#if GM_SIZEOF_UP_T == 4
  gm_u32_n_t page_port_pad;
#endif 
  gm_up_n_t page_port;		/* also stores port */ 
  /* 8 */ 
#if GM_SIZEOF_DP_T == 4
  gm_u32_n_t _packed_dma_addr_pad;
#endif 
  gm_dp_n_t _packed_dma_addr;	/* also stores reference count */ 
}

gm_pte_t;

/* Fake port numbers to identify fake PTEs, which are used to store other
   types of information. */

#define GM_GATEWAY_CACHE_ENTRY_MAGIC_PORT_ID GM_NUM_PORTS
#define GM_CONNECTION_HASH_ENTRY_MAGIC_PORT_ID (GM_NUM_PORTS + 1)

/* Fake PTE for caching gateways. */

typedef struct gm_gateway_cache_entry 
{
  gm_ethernet_mac_addr_t destination;
  gm_u16_n_t magic;		/* ensures will not match any PTE */
  /* 8 */
  gm_u32_n_t valid_time;
  gm_u16_n_t gateway_node_id;
  gm_u16_n_t pad;
} gm_gateway_cache_entry_t;

/* Fake PTE for hashing connections. */

typedef struct 
{
  gm_u32_n_t global_id;
  gm_u32_n_t magic;
  /* 8 */
  gm_connection_lp_t connection;
  gm_u32_n_t pad;
} gm_connection_hash_entry_t;

/* A cached version of a gm_pte_n_t. 2^N bytes. */
typedef union gm_cached_pte
{
  gm_gateway_cache_entry_t gce;
  gm_connection_hash_entry_t che;
  gm_pte_t pte;
}

gm_cached_pte_t;

/* A cache of a host-resident page hash table.  Includes a ROOT for
   the list of recently used cache entries, an array of bins in which
   to store ENTRYs, and a CNT of the number of cached entries. */
typedef struct gm_page_hash_cache
{
  gm_u32_n_t cnt;
  gm_u32_n_t gateway_cnt;
  /* 8 */
  /* extra entry at end to allow wraparound optimization. */
  gm_cached_pte_lp_t const entry;
  gm_u32_n_t max_index;
}

gm_page_hash_cache_t;

/* this structure is used to pass the args of a directcopy to the kernel */
typedef struct gm_directcopy
{
  /* virtual address of the source buffer in the sender process */
  gm_up_t source_addr;
  /* virtual address of the target buffer on the receiver process */
  gm_up_t target_addr;
  /* length of the buffer to copy */
  gm_size_t length;
  /* port id and board id to identify the source process (GET protocol) */
  gm_u16_t source_port_id;
  gm_u16_t source_instance_id;
}

gm_directcopy_t;

typedef struct gm_firmware_log_entry
{
  gm_lp_n_t message;
  gm_s32_n_t time;
}

gm_firmware_log_entry_t;

/**********************************************************************
 * Packet-related types
 **********************************************************************
 All fields of GM packets use network byte order, unless otherwise noted. */

/* The header of a GM data segment sent over the network. */
				/* 6 words */
typedef struct gm_packet_header
{
  gm_u16_n_t type;
  gm_u16_n_t subtype;

  gm_u32_n_t target_global_id;
  /* 8 */
  gm_sexno_t sexno;

  gm_u16_n_t length;		/* of payload */
  gm_u8_n_t target_subport_id;
  gm_u8_n_t sender_subport_id;
  /* 8 */
  gm_u32_n_t sender_global_id;
  gm_u16_n_t connection_id_on_target; /* might be wrong */
  gm_u16_n_t connection_id_on_sender;
}

gm_packet_header_t;

#define GM_HEADER_PAYLOAD(p) ((gm_u8_n_t *)((gm_packet_header_t *)(p)+1))

/* A GM data segment consists of a header and a payload of GM_MTU
   bytes or less. */
typedef struct gm_packet
{
  gm_packet_header_t header;
  gm_u8_n_t payload[GM_MTU];
}

gm_packet_t;

/* A GM data segment used for a put.  All fields must be
   worst-case for all GM implementations for compatibility. */
typedef struct gm_put_packet
{
  gm_packet_header_t header;
  /* 8 */
  gm_remote_ptr_n_t slave_addr;
  /* 8 */
  gm_u8_n_t payload[GM_MTU + 8];
}

gm_put_get_packet_t;

/* A GM ack packet */
typedef struct gm_ack_packet	/* 4 words */
{
  gm_u16_n_t type;
  gm_u16_n_t subtype;

  gm_u32_n_t target_global_id;
  /* 8 */
  gm_sexno_t sexno;

  gm_s32_n_t reserved;
  /* 8 */
  gm_u32_n_t sender_global_id;
  gm_u16_n_t connection_id_on_target; /* always right */
  gm_u16_n_t connection_id_on_sender;
}

gm_ack_packet_t;

/****************
 * Recv events
 ****************/

/* Make sure the assumptions made by the following typedefs are legit */

#if GM_SIZEOF_UP_T > 8 || GM_MAX_DMA_GRANULARITY > 8
#error broken typedefs
#endif

#if GM_RDMA_GRANULARITY < 32

#error possibly broken typedefs
#endif


				/* Struct including the minimum data
				   that needs to be DMAd for a normal
				   receive, and having GM_RDMA_GRANULARITY. */
struct _gm_recv_event
{
  /* Pad to GM_RDMA_GRANULARITY bytes */
#if 24 % GM_RDMA_GRANULARITY
  gm_u8_n_t _reserved[GM_RDMA_GRANULARITY - 24 % GM_RDMA_GRANULARITY];
#endif
  /* 8 */
#if GM_SIZEOF_UP_T == 4
  gm_u32_n_t buffer_pad;
#endif
  gm_up_n_t buffer;
  /* 8 */
  gm_u32_n_t reserved_after_buffer;
  gm_u32_n_t length;
  /* 8 */
  gm_u16_n_t sender_node_id;
  gm_u16_n_t reserved_after_sender_node_id;
  gm_u8_n_t tag;
  gm_u8_n_t size;
  gm_u8_n_t sender_port_id;
  gm_u8_n_t type;
};

				/* Struct holding only the information
				   needed for a fast receive with NO
				   GRANULARITY PADDING. */
struct _gm_fast_recv_event
{
#if GM_SIZEOF_UP_T == 4
  gm_u32_n_t message_pad;
#endif
  gm_up_n_t message;
  /* 8 */
#if GM_SIZEOF_UP_T == 4
  gm_u32_n_t buffer_pad;
#endif
  gm_up_n_t buffer;
  /* 8 */
  gm_u32_n_t reserved_after_buffer;
  gm_u32_n_t length;
  /* 8 */
  gm_u16_n_t sender_node_id;
  gm_u16_n_t reserved_after_sender_node_id;
  gm_u8_n_t tag;
  gm_u8_n_t size;
  gm_u8_n_t sender_port_id;
  gm_u8_n_t type;
};


				/* Struct holding only the information
				   needed to be DMAd to report an
				   ethernet recv. */
struct _gm_ethernet_recv_event
{
  /* Pad to GM_RDMA_GRANULARITY bytes */
#if (16 + GM_SIZEOF_UP_T) % GM_RDMA_GRANULARITY
  gm_u8_n_t _reserved[GM_RDMA_GRANULARITY
		      - (16 + GM_SIZEOF_UP_T) % GM_RDMA_GRANULARITY];
#endif
  gm_up_n_t buffer;
  /* 8 */
  gm_u16_n_t ip_checksum;	/* IPv4 partial checksum, or checksum
				   of entire packet; flags tell which */
  gm_u16_n_t ip_header_len;	/* future use */
  gm_u32_n_t length;
  /* 8 */
  gm_u16_n_t sender_node_id;
  gm_u8_n_t flags;
  gm_u8_n_t reserved_after_flags;
  gm_u8_n_t tag;
  gm_u8_n_t size;
  gm_u8_n_t sender_port_id;
  gm_u8_n_t type;
};

typedef struct gm_recv_queue_slot
{
  gm_u8_n_t _reserved[2 * GM_MAX_FAST_RECV_BYTES - sizeof (gm_recv_event_t)];
  gm_recv_event_t event;
}

gm_recv_queue_slot_t;
#define GM_SIZEOF_GM_RECV_QUEUE_SLOT_T (2 * GM_MAX_FAST_RECV_BYTES)

/* *INDENT-OFF* */
/* */ GM_TOP_LEVEL_ASSERT (sizeof (gm_recv_queue_slot_t)
			   == GM_SIZEOF_GM_RECV_QUEUE_SLOT_T);
/* *INDENT-ON* */

typedef union gm_myrinet_payload
{
  struct
  {
    gm_u16_n_t type;
    char payload[GM_MYRINET_MTU-2];
  }
  as_myrinet;
  gm_packet_t as_gm;
  gm_put_get_packet_t as_put_get_packet;
  char as_bytes[1];
  /* FIXME: remove this */
  char as_ethernet[GM_ROUNDUP (u32, GM_ETHERNET_OVER_MYRINET_MTU, 8)];
} gm_myrinet_payload_t;

/****
 * Token related types
 ****/

enum gm_send_event_type
{
  GM_NO_SEND_EVENT = 0,
  GM_SET_ALARM_EVENT = 1,
  GM_FLUSH_ALARM_EVENT = 2,
  GM_RAW_SEND_EVENT = 3,
  /*  xxxGM_UNRELIABLE_SEND_EVENT = 4, */
  GM_WAKE_REQUEST_EVENT = 5,

  /* Special steamlined send types for
     messages less than 256 bytes
     long. */

  GM_FAST_SEND_EVENT_0 = 6, GM_FAST_SEND_EVENT_1 = 7,
  GM_FAST_SEND_EVENT_2 = 8, GM_FAST_SEND_EVENT_3 = 9,
  GM_FAST_SEND_EVENT_4 = 10, GM_FAST_SEND_EVENT_5 = 11,
  GM_FAST_SEND_EVENT_6 = 12, GM_FAST_SEND_EVENT_7 = 13,
  GM_FAST_SEND_EVENT_8 = 14, GM_FAST_SEND_EVENT_9 = 15,
  GM_FAST_SEND_EVENT_10 = 16, GM_FAST_SEND_EVENT_11 = 17,
  GM_FAST_SEND_EVENT_12 = 18, GM_FAST_SEND_EVENT_13 = 19,
  GM_FAST_SEND_EVENT_14 = 20, GM_FAST_SEND_EVENT_15 = 21,
  GM_FAST_SEND_EVENT_16 = 22, GM_FAST_SEND_EVENT_17 = 23,
  GM_FAST_SEND_EVENT_18 = 24, GM_FAST_SEND_EVENT_19 = 25,
  GM_FAST_SEND_EVENT_20 = 26, GM_FAST_SEND_EVENT_21 = 27,
  GM_FAST_SEND_EVENT_22 = 28, GM_FAST_SEND_EVENT_23 = 29,
  GM_FAST_SEND_EVENT_24 = 30, GM_FAST_SEND_EVENT_25 = 31,
  GM_FAST_SEND_EVENT_26 = 32, GM_FAST_SEND_EVENT_27 = 33,
  GM_FAST_SEND_EVENT_28 = 34, GM_FAST_SEND_EVENT_29 = 35,
  GM_FAST_SEND_EVENT_30 = 36, GM_FAST_SEND_EVENT_31 = 37,

  GM_FAST_SEND_HIGH_EVENT_0 = 38, GM_FAST_SEND_HIGH_EVENT_1 = 39,
  GM_FAST_SEND_HIGH_EVENT_2 = 40, GM_FAST_SEND_HIGH_EVENT_3 = 41,
  GM_FAST_SEND_HIGH_EVENT_4 = 42, GM_FAST_SEND_HIGH_EVENT_5 = 43,
  GM_FAST_SEND_HIGH_EVENT_6 = 44, GM_FAST_SEND_HIGH_EVENT_7 = 45,
  GM_FAST_SEND_HIGH_EVENT_8 = 46, GM_FAST_SEND_HIGH_EVENT_9 = 47,
  GM_FAST_SEND_HIGH_EVENT_10 = 48, GM_FAST_SEND_HIGH_EVENT_11 = 49,
  GM_FAST_SEND_HIGH_EVENT_12 = 50, GM_FAST_SEND_HIGH_EVENT_13 = 51,
  GM_FAST_SEND_HIGH_EVENT_14 = 52, GM_FAST_SEND_HIGH_EVENT_15 = 53,
  GM_FAST_SEND_HIGH_EVENT_16 = 54, GM_FAST_SEND_HIGH_EVENT_17 = 55,
  GM_FAST_SEND_HIGH_EVENT_18 = 56, GM_FAST_SEND_HIGH_EVENT_19 = 57,
  GM_FAST_SEND_HIGH_EVENT_20 = 58, GM_FAST_SEND_HIGH_EVENT_21 = 59,
  GM_FAST_SEND_HIGH_EVENT_22 = 60, GM_FAST_SEND_HIGH_EVENT_23 = 61,
  GM_FAST_SEND_HIGH_EVENT_24 = 62, GM_FAST_SEND_HIGH_EVENT_25 = 63,
  GM_FAST_SEND_HIGH_EVENT_26 = 64, GM_FAST_SEND_HIGH_EVENT_27 = 65,
  GM_FAST_SEND_HIGH_EVENT_28 = 66, GM_FAST_SEND_HIGH_EVENT_29 = 67,
  GM_FAST_SEND_HIGH_EVENT_30 = 68, GM_FAST_SEND_HIGH_EVENT_31 = 69,

  GM_SEND_EVENT_0 = 70, GM_SEND_EVENT_1 = 71, GM_SEND_EVENT_2 = 72,
  GM_SEND_EVENT_3 = 73, GM_SEND_EVENT_4 = 74, GM_SEND_EVENT_5 = 75,
  GM_SEND_EVENT_6 = 76, GM_SEND_EVENT_7 = 77, GM_SEND_EVENT_8 = 78,
  GM_SEND_EVENT_9 = 79, GM_SEND_EVENT_10 = 80, GM_SEND_EVENT_11 = 81,
  GM_SEND_EVENT_12 = 82, GM_SEND_EVENT_13 = 83, GM_SEND_EVENT_14 = 84,
  GM_SEND_EVENT_15 = 85, GM_SEND_EVENT_16 = 86, GM_SEND_EVENT_17 = 87,
  GM_SEND_EVENT_18 = 88, GM_SEND_EVENT_19 = 89, GM_SEND_EVENT_20 = 90,
  GM_SEND_EVENT_21 = 91, GM_SEND_EVENT_22 = 92, GM_SEND_EVENT_23 = 93,
  GM_SEND_EVENT_24 = 94, GM_SEND_EVENT_25 = 95, GM_SEND_EVENT_26 = 96,
  GM_SEND_EVENT_27 = 97, GM_SEND_EVENT_28 = 98, GM_SEND_EVENT_29 = 99,
  GM_SEND_EVENT_30 = 100, GM_SEND_EVENT_31 = 101,

  GM_PUT_SEND_EVENT = 102,

  GM_GET_SEND_EVENT = 103,
  GM_RESUME_SENDING_EVENT = 104,
  GM_DROP_SENDS_EVENT = 105,
  GM_ETHERNET_SEND_EVENT = 106,
  GM_GATEWAY_SEND_EVENT = 107,
  GM_ETHERNET_MARK_AND_SEND_EVENT_XXX = 108,
  GM_ETHERNET_MARK_AND_BROADCAST_EVENT_XXX = 109,
  GM_TRACE_START_EVENT = 110,
  GM_TRACE_STOP_EVENT = 111,

  GM_DATAGRAM_SEND_EVENT_0, GM_DATAGRAM_SEND_EVENT_1,
  GM_DATAGRAM_SEND_EVENT_2, GM_DATAGRAM_SEND_EVENT_3,
  GM_DATAGRAM_SEND_EVENT_4, GM_DATAGRAM_SEND_EVENT_5,
  GM_DATAGRAM_SEND_EVENT_6, GM_DATAGRAM_SEND_EVENT_7,
  GM_DATAGRAM_SEND_EVENT_8, GM_DATAGRAM_SEND_EVENT_9,
  GM_DATAGRAM_SEND_EVENT_10, GM_DATAGRAM_SEND_EVENT_11,
  GM_DATAGRAM_SEND_EVENT_12, GM_DATAGRAM_SEND_EVENT_13,
  GM_DATAGRAM_SEND_EVENT_14, GM_DATAGRAM_SEND_EVENT_15,
  GM_DATAGRAM_SEND_EVENT_16, GM_DATAGRAM_SEND_EVENT_17,
  GM_DATAGRAM_SEND_EVENT_18, GM_DATAGRAM_SEND_EVENT_19,
  GM_DATAGRAM_SEND_EVENT_20, GM_DATAGRAM_SEND_EVENT_21,
  GM_DATAGRAM_SEND_EVENT_22, GM_DATAGRAM_SEND_EVENT_23,
  GM_DATAGRAM_SEND_EVENT_24, GM_DATAGRAM_SEND_EVENT_25,
  GM_DATAGRAM_SEND_EVENT_26, GM_DATAGRAM_SEND_EVENT_27,
  GM_DATAGRAM_SEND_EVENT_28, GM_DATAGRAM_SEND_EVENT_29,
  GM_DATAGRAM_SEND_EVENT_30, GM_DATAGRAM_SEND_EVENT_31,

  GM_PIO_DATAGRAM_4_SEND_EVENT,
  GM_PIO_DATAGRAM_8_SEND_EVENT,
  GM_PIO_DATAGRAM_12_SEND_EVENT,
  GM_PIO_DATAGRAM_16_SEND_EVENT
};

#define GM_PRIORITY_SIZE(priority,size) ((priority)<<5|size)
#define GM_PRIORITY_SIZE__PRIORITY(ps) ((ps)>>5)
#define GM_PRIORITY_SIZE__SIZE(ps) ((ps)&31)

union gm_interrupt_descriptor
{
  volatile gm_u32_n_t type;
  struct
  {
    volatile gm_u32_n_t type;
    volatile gm_u32_n_t port;
  }
  wake;
  struct
  {
    volatile gm_u32_n_t type;
    volatile gm_lp_n_t string;
  }
  print;
  struct
  {
    volatile gm_u32_n_t type;
    volatile gm_lp_n_t file;
    volatile gm_u32_n_t line;
    volatile gm_lp_n_t text;
  }
  failed_assertion;
  struct
  {
    volatile gm_u32_n_t type;
    volatile gm_u32_n_t len;
    volatile gm_u16_n_t checksum;	/* 0 if unknown */
    gm_u8_n_t reserved_after_checksum[6];
  }
  ethernet_recv;
  struct
  {
    volatile gm_u32_n_t type;
    gm_u32_n_t length;
    gm_lp_n_t buffer;
  }
  write;
  struct
  {
    volatile gm_u32_n_t type;
    gm_u32_n_t port;
    gm_up_n_t pointer;
    gm_u32_n_t repetitions;
  }
  bogus;
  struct
  {
    volatile gm_u32_n_t type;
    gm_u32_n_t node_id;
    gm_u32_n_t global_id;
  }
  node_id_translation;
  struct
  {
    gm_u8_n_t _reserved[23];	/* make size of union be 24 bytes */
    gm_u8_n_t ready;
  }
  last_byte;
};

/****************************************************************
 * Send events
 ****************************************************************/

/* Special care must be taken when laying out send events to ensure
   that the following conditions are satisfied:
   o The 1-byte TYPE field must be last.
   o All fields must be aligned assuming that the END of the event
     is aligned on an 8-byte boundary.
   o The structures must have no implicit padding on the end.  This
     means if the largest field is N bytes, then the TYPE field must
     be just below a N-byte boundary. */

#if GM_SIZEOF_UP_T == 4
#define GM_UP_T(name) gm_u32_t reserved_before_ ## name ; gm_up_t name
#else
#define GM_UP_T(name) gm_up_t name
#endif

struct gm_send_send_event	/* 16 bytes */
{
#if GM_SIZEOF_UP_T == 4
  gm_u32_t pad_before_message;
#endif
  gm_up_n_t message;
  /* 8 */
  gm_u32_n_t length;
  gm_u16_n_t target_node_id;
  gm_u8_n_t target_subport_id;
  gm_u8_n_t type;
};

struct gm_pio_datagram_send_event /* 8 bytes */
{
  /* 8 */
  gm_u32_n_t data;
  gm_u16_n_t target_node_id;
  gm_u8_n_t target_subport_id;
  gm_u8_n_t type;
};

struct gm_put_get_send_event	/* 24 bytes */
{
#if GM_SIZEOF_UP_T == 4
  gm_u32_t pad_before_local_buffer;
#endif
  gm_up_n_t local_buffer;
  /* 8 */
  gm_remote_ptr_n_t remote_buffer;
  /* 8 */
  gm_u32_n_t length;
  gm_u16_n_t remote_node_id;
  gm_u8_n_t remote_subport_id;
  gm_u8_n_t type;
};

struct gm_fast_send_send_event	/* 8 or 16 bytes */
{
#if GM_SIZEOF_UP_T == 8
  /* 8 */
  volatile gm_up_n_t message;
  /* 8 */
  gm_u32_n_t reserved;
  gm_u16_n_t target_node_id;
  gm_u8_n_t length;
  gm_u8_n_t type;
#elif GM_SIZEOF_UP_T == 4
  /* 8 */
  volatile gm_up_n_t message;
  gm_u16_n_t target_node_id;
  gm_u8_n_t length;
  gm_u8_n_t type;
#endif
};

struct gm_raw_send_send_event	/* 24 bytes */
{
#if GM_SIZEOF_UP_T == 4
  gm_u32_t pad_before_message;
#endif
  gm_up_n_t message;
  /* 8 */
  gm_u32_n_t total_length;
  gm_u32_n_t route_length;
  /* 8 */
  gm_u32_n_t reserved_after_route_length;
  gm_u16_n_t cleared;
  gm_u8_n_t reserved_after_cleared;
  gm_u8_n_t type;
};

struct gm_set_alarm_send_event	/* 16 bytes */
{
  /* 8 */
  gm_u64_n_t usecs;
  /* 8 */
  gm_u8_n_t _reserved_after_usecs[7];
  gm_u8_n_t type;
};

struct gm_simple_send_event	/* 4 bytes */
{
  gm_u8_n_t reserved[3];
  gm_u8_n_t type;
};

struct gm_probe_send_event	/* 4 bytes */
{
  gm_u16_n_t target_node_id;
  gm_u8_n_t _reserved_after_target_port_id;
  gm_u8_n_t type;
};

struct gm_resend_send_event	/* 4 bytes */
{
  gm_u16_n_t target_node_id;
  gm_u8_n_t target_subport_id;
  gm_u8_n_t type;
};

struct gm_wake_request_send_event
{
  gm_u16_n_t recv_slot;
  gm_u8_n_t reserved;
  gm_u8_n_t type;
};

typedef union gm_unique_id_64_t	/* 8 bytes */
{
  gm_u64_t as_unswapped_u64;	/* No endian swapping checking here */
  gm_u8_n_t as_bytes[6];
}

gm_unique_id_64_t;

#define GM_MAX_ETHERNET_IMMEDIATE_GATHER_CNT \
(24 / (sizeof (gm_dp_t) + sizeof (gm_u16_t)))

struct gm_ethernet_send_event
{
  gm_u8_t packed_gather_list[GM_SIZEOF_SEND_QUEUE_SLOT - 8];
  /* 8 */
  union 
  {
    /* Details of ethernet sends destined for GM nodes. */
    struct 
    {
      gm_u8_n_t reserved[4];
      gm_u16_n_t target_node_id;
    } node;			/* GM_GATEWAY_SEND_EVENT */
    /* Details of ethernet sends through gateways. */
    struct
    {
      gm_ethernet_mac_addr_t target_mac_addr;
    } gateway;			/* GM_ETHERNET_SEND_EVENT */
  } via;
  gm_u8_n_t gather_cnt;
  gm_u8_n_t type;
};

/* A data structure used to pass messages from the Host to the LANai.
   All fields use LANai (network) byte order except as noted.  */
struct gm_send_queue_slot
{
  gm_u8_n_t as_bytes[GM_SIZEOF_SEND_QUEUE_SLOT - 1];
  gm_u8_n_t type;
};

/* *INDENT-OFF* */
/* */ GM_TOP_LEVEL_ASSERT (sizeof (struct gm_send_queue_slot)
			   == GM_SIZEOF_SEND_QUEUE_SLOT);
/* *INDENT-ON* */

#define GM_ST_ACKABLE_P(st) (st->common.type == GM_ST_RELIABLE		\
			     || st->common.type == GM_ST_DIRECTED	\
			     || st->common.type == GM_ST_PROBE)

#define GM_SEND_QUEUE_SLOT_EVENT(sqs, type)				\
((struct gm_ ## type ## _send_event *)(sqs+1)-1)


/* The state of a subport. */
typedef struct gm_subport
{
  gm_subport_lp_t next;
  gm_subport_lp_t prev;
  gm_connection_lp_t connection;

  gm_send_token_lp_t first_send_token;
  gm_send_token_lp_t last_send_token;

  gm_s32_n_t reserved;
  gm_u8_n_t id;
  gm_u8_n_t disabled;
  gm_u8_n_t reserved_after_disabled[2];

  /* We only store the 16 MSbs of the
     progress time, since we only use it
     to check for fatal timeouts, which
     are huge infrequent */
  gm_s32_n_t progress_time;
}

gm_subport_t;

/* A data structure used to queue sends internally in the LANai. */
typedef union gm_send_token
{
  struct gm_st_common
  {
    gm_send_token_lp_t next;
    gm_u8_n_t type;
    gm_s8_n_t _reserved_after_type[3];
    /* 8 */
    gm_u32_n_t sendable;	/* Must be nonzero for all types that
				   may be sent */
    gm_subport_lp_t subport;
  }
  common;
  struct gm_st_ackable
  {
    gm_send_token_lp_t next;
    gm_u8_n_t type;
    gm_s8_n_t _reserved_after_type;
    gm_u16_n_t target_subport_id;
    /* 8 */
    gm_u32_n_t send_len;
    gm_subport_lp_t subport;
    /* 8 */
    gm_up_n_t orig_ptr;
    gm_up_n_t send_ptr;
  }
  ackable;
  struct gm_st_reliable
  {
    gm_send_token_lp_t next;
    gm_u8_n_t type;
    gm_s8_n_t size;
    gm_u16_n_t target_subport_id;
    /* 8 */
    gm_u32_n_t send_len;
    gm_subport_lp_t subport;
    /* 8 */
    gm_up_n_t orig_ptr;
    gm_up_n_t send_ptr;
#if GM_FAST_SMALL_SEND
    gm_lp_n_t data;
#endif
  }
  reliable;
  struct gm_st_put_get
  {
    gm_send_token_lp_t next;
    gm_u8_n_t type;
    gm_s8_n_t reserved__was_size; /* HACK */
    gm_u16_n_t remote_subport_id;
    /* 8 */
    gm_u32_n_t length;
    gm_subport_lp_t subport;
    /* 8 */
    gm_up_n_t orig_local_ptr;
    gm_up_n_t local_ptr;
    /* 8 */
    gm_remote_ptr_n_t remote_ptr;
  }
  put_get;
  struct gm_st_raw
  {
    gm_send_token_lp_t next;
    gm_u8_n_t type;
    gm_s8_n_t size;		/* HACK */
    gm_u16_n_t target_subport_id;	/* HACK */
    /* 8 */
    gm_u32_n_t total_length;
    gm_subport_lp_t subport;
    /* 8 */
    gm_up_n_t orig_ptr;
    gm_up_n_t send_ptr;		/* HACK */
    gm_u32_n_t route_length;
    gm_u32_n_t reserved_after_route_length;
  }
  raw;
  struct gm_st_probe
  {
    gm_send_token_lp_t next;
    gm_u8_n_t type;
    gm_s8_n_t _reserved_after_type;
    gm_u16_n_t target_subport_id;
    /* 8 */
    gm_u32_n_t send_len;
    gm_subport_lp_t subport;
  }
  probe;
  struct gm_st_ethernet
  {
    gm_send_token_lp_t next;
    gm_u8_n_t type;
    gm_u8_n_t saved_event_type;
    gm_u8_n_t serial_number;
    gm_u8_n_t _reserved_after_type;
    /* 8 */
    gm_ethernet_send_event_lp_t event;/* sendable */
    gm_subport_lp_t subport;
  }
  ethernet;
  struct gm_st_datagram
  {
    gm_send_token_lp_t next;
    gm_u8_n_t type;
    gm_s8_n_t size;
    gm_u16_n_t target_subport_id;
    /* 8 */
    gm_u32_n_t send_len;
    gm_subport_lp_t subport;
    /* 8 */
    gm_up_n_t send_ptr;
#if GM_SIZEOF_UP_T == 4
    gm_u32_t pad_after_send_ptr;
#endif
  }
  datagram;
  struct gm_st_pio_datagram
  {
    gm_send_token_lp_t next;
    gm_u8_n_t type;
    gm_s8_n_t size;
    gm_u16_n_t target_subport_id;
    /* 8 */
    gm_u32_n_t send_len;
    gm_subport_lp_t subport;
    /* 8 */
    gm_u32_n_t data[4];		/* up to 16 bytes of data */
  }
  pio_datagram;
}

gm_send_token_t;

/* Structure to keep track of unacked sends.
   There should be at least as many send records as send tokens. */
typedef struct gm_send_record
{
  gm_send_record_lp_t next;
  gm_send_token_lp_t send_token;
  /* 8 */
  gm_up_n_t before_ptr;
#if GM_SIZEOF_UP_T != 8
  gm_sexno_t pad;
#endif
  /* 8 */
  gm_u8_n_t requires_explicit_ack;
  gm_u8_n_t reserved[7];
  /* 8 */
  gm_u32_n_t before_len;
  gm_s32_n_t resend_time;
  /* 8 */
  gm_sexno_t sexno;
  gm_u32_n_t packet_len;
}

gm_send_record_t;

/****************************************************************
 * Host recv tokens
 ****************************************************************/

/* Structure used to pass receive tokens (which specify locations in
   which messages may be received) from the host to the LANai. */
typedef struct gm_host_recv_token
{
  gm_up_n_t volatile message;
#if GM_SIZEOF_UP_T == 8
  gm_u32_n_t _reserved1;
#endif
  gm_u8_n_t volatile tag;
  gm_u8_n_t volatile priority;
  gm_u8_n_t volatile size;
  gm_u8_n_t volatile ready;	/* must be last */
}

gm_host_recv_token_t;

/* *INDENT-OFF* */
/* */ GM_TOP_LEVEL_ASSERT (sizeof (gm_host_recv_token_t)
			   == GM_SIZEOF_HOST_RECV_TOKEN);
/* *INDENT-ON* */

typedef union gm_recv_token_key
{
  struct gm_recv_token_key_parts
  {
    gm_u16_n_t sender_node_id;
    gm_u8_n_t sender_subport_id;
    gm_u8_n_t target_subport_id;
  }
  parts;
  gm_u32_n_t whole;
}

gm_recv_token_key_t;

/* Structure stored at the end of a receive queue slot if a message is
   too big to fit directly in the recv queue slot. */
typedef struct gm_recv_token
{
  gm_recv_token_key_t key;
  gm_u8_n_t size;
  gm_u8_n_t tag;
  gm_u16_n_t reserved_after_tag;
  /* 8 */
  gm_recv_token_lp_t next;
  gm_u32_n_t _reserved_after_next;
  /* 8 */
  gm_up_n_t orig_ptr;
  gm_up_n_t recv_ptr;
  /* 8 */
  /* Special field for puts. */
  gm_u8_n_t start_copy_bytes[GM_MAX_DMA_GRANULARITY];
#if GM_MAX_DMA_GRANULARITY != 8
  gm_u8_n_t reserved_after_start_copy_bytes[8 - GM_MAX_DMA_GRANULARITY];
#endif
}

gm_recv_token_t;

/* State of the connection to a remote node. There is only one
   connection to each remote node, but multiple streams of data are
   multiplexed over each connection, providing reliable ordered
   delivery between subports.

   Much of the the connection state is stored directly in the
   ACK_MESSAGE.  */

#define GM_CONNECTION_FIELDS						\
    /* Fields used to form a doubly-linked list of connections with	\
       outstanding sends. */						\
    gm_connection_lp_t next_active;					\
    gm_connection_lp_t prev_active;					\
    /* 8 */								\
    gm_u32_n_t active_subport_bitmask;	/* 0 if inactive */		\
    gm_u8_n_t probable_crc_error_cnt;					\
    gm_u8_n_t misrouted_packet_error_cnt;				\
    gm_u8_n_t has_route;						\
    gm_u8_n_t route_len;						\
    /* 8 */								\
    /* Ack prestaging area: ROUTE must preceed ACK_PACKET */		\
    gm_u8_n_t route[GM_MAX_NETWORK_DIAMETER];	/* 4 words */		\
    /* 8 */								\
    gm_ack_packet_t ack_packet;	/* 4 words */				\
    /* 8 */								\
    /* end of ack prestaging area */					\
    gm_sexno_t send_sexno;						\
    /* routes that are not refreshed by the mapper are cleared later*/	\
    gm_s8_n_t retired;							\
    /* When resending, scan the entire send record list for resends. */ \
    gm_s8_n_t resending;						\
    gm_s8_n_t outstanding_get_cnt;					\
    gm_u8_n_t node_type;						\
    /* 8 */								\
    gm_send_record_lp_t first_send_record;				\
    gm_send_record_lp_t last_send_record;				\
    /* 8 */								\
    /* may be nonzero even if inactive */				\
    gm_subport_lp_t first_active_send_port;				\
    gm_sexno_t close_sexno;						\
    /* 8 */								\
    gm_s64_n_t open_time;						\
    /* 8 */								\
    gm_s64_n_t close_time;						\
    /* 8 */								\
    gm_s32_n_t known_alive_time;					\
    gm_u32_n_t bad_crc_cnt;						\
    /* 8 */								\
    gm_myrinet_packet_descriptor_free_list_t ack_descriptor_free_list;	\
    /* 8 */								\
    gm_myrinet_packet_descriptor_t ack_descriptor;			\
    /* 8 */								\
    gm_s32_t yp_blackout_time;						\
    gm_u16_t ethernet_seqno;						\
    gm_u16_t reserved

struct gm_connection_fields
{
  GM_CONNECTION_FIELDS;
};

typedef struct gm_connection
{
  GM_CONNECTION_FIELDS;
#if GM_MIN_SUPPORTED_SRAM > 256
  GM_POW2_PAD (sizeof (struct gm_connection_fields));
#endif /* GM_MIN_SUPPORTED_SRAM > 25 */
}

gm_connection_t;

/* These macros work both from the LANai and host, since they only
   operate on bytes. */

#define GM_CONNECTION_HAS_ROUTE(c) ((c)->has_route)
#define GM_CONNECTION_ROUTE(c) (&(c)->route[GM_MAX_NETWORK_DIAMETER	\
					   - gm_ntohc ((c)->route_len)])
#define GM_CONNECTION_SET_HAS_ROUTE(c,b) ((c)->has_route = -1)
#define GM_CONNECTION_ROUTE_LEN(c) ((c)->has_route?(c)->route_len:GM_NO_ROUTE)
#define GM_CONNECTION_CLEAR_ROUTE(c) do {				\
  (c)->has_route = 0;							\
  (c)->route_len = 0;							\
  GM_STBAR ();								\
} while (0)
/* Be careful to ensure the route length is zero while setting the route. */
#define GM_CONNECTION_SET_ROUTE(c,len,ptr) do {				\
  GM_CONNECTION_CLEAR_ROUTE(c);						\
  if (len != GM_NO_ROUTE)						\
    {									\
      {									\
	char *__from, *__to, *__limit;					\
									\
	__from = (ptr);							\
	__to = (char *) &(c)->route[GM_MAX_NETWORK_DIAMETER-(len)];	\
	__limit = (char *) &(c)->route[GM_MAX_NETWORK_DIAMETER];	\
	while (__to < __limit)						\
	  *__to++ = *__from++;						\
      }									\
      GM_STBAR ();							\
      (c)->route_len = (len);						\
      (c)->has_route = -1;						\
    }									\
} while (0)

#define GM_CONNECTING 1
#define GM_CONNECTED 0

/* The LANai-resident queues and other data for a user port that the
   user can modify directly.

   NOTE: This structure is mapped into user space, so no assumptions
   about the integrity of this data should be made. */

typedef struct gm_port_unprotected_lanai_side
{
  /* Host<->LANai token queues */
  struct gm_send_queue_slot send_token_queue[GM_NUM_SEND_QUEUE_SLOTS];
  /* 8 */
  /* Extra slot for wraparound. */
  gm_host_recv_token_t recv_token_queue[GM_NUM_RECV_TOKEN_QUEUE_SLOTS + 1];
#if GM_MCP
  gm_u8_n_t padding[GM_PORT_MAPPED_SRAM_LEN
		    - ((GM_NUM_SEND_QUEUE_SLOTS
			* sizeof (struct gm_send_queue_slot))
		       + ((GM_NUM_RECV_TOKEN_QUEUE_SLOTS + 1)
			  * sizeof (gm_host_recv_token_t)))];
#endif
}

gm_port_unprotected_lanai_side_t;

#define GM_PORT_PROTECTED_LANAI_SIDE_FIELDS				 \
    gm_send_queue_slot_lp_t send_token_queue_slot;      /*lanai addr */	 \
    gm_host_recv_token_lp_t recv_token_queue_slot;      /*lanai addr */	 \
    /* 8 */								 \
    gm_u8_n_t wake_host;						 \
    gm_u8_n_t open;							 \
    gm_u8_n_t alarm_set;						 \
    gm_u8_n_t enable_nack_down_flag;					 \
    gm_u32_n_t privileged;						 \
    /* 8 */								 \
    /* lanai internal queues */						 \
    gm_send_token_lp_t first_free_send_token;	/*lanai addr */		 \
    gm_send_token_lp_t last_free_send_token;	/*lanai addr */		 \
    /* 8 */								 \
    gm_send_token_t _send_tokens[GM_NUM_SEND_QUEUE_SLOTS /*really!*/ ];	 \
    /* 8 */								 \
    gm_recv_token_t _recv_tokens[GM_NUM_RECV_TOKENS];			 \
    /* 8 */								 \
    gm_recv_token_lp_t free_recv_token[GM_NUM_PRIORITIES][GM_NUM_SIZES]; \
    /* 8 */								 \
    /* The LANai->host recv queue */					 \
    gm_recv_token_lp_t free_recv_tokens;	/*lanai addr */		 \
    _gm_sent_token_report_lp_t sent_slot;				 \
    /* 8 */								 \
    gm_port_protected_lanai_side_lp_t next_with_alarm;			 \
    gm_u32_n_t recv_queue_slot_num;					 \
    /* 8 */								 \
    /* Assume page size is the worst-case here */			 \
    gm_dp_n_t recv_queue_slot_dma_addr [GM_NUM_RECV_QUEUE_SLOTS		 \
				     + GM_NUM_RECV_QUEUE_SLOTS%2];	 \
    /* 8 */								 \
    gm_up_n_t recv_queue_slot_host_addr [GM_NUM_RECV_QUEUE_SLOTS	 \
				      + GM_NUM_RECV_QUEUE_SLOTS%2];	 \
    /* 8 */								 \
    gm_port_protected_lanai_side_lp_t next_with_sent_packets;		 \
    gm_u32_n_t active_subport_cnt;					 \
    /* 8 */								 \
    /* Staging area for GM_SENT_EVENT events. */			 \
    gm_tokens_sent_t sent;						 \
    /* 8 */								 \
    gm_s32_n_t alarm_time;						 \
    gm_u32_n_t id;							 \
    /* 8 */								 \
    gm_u32_n_t unacceptable_recv_sizes[GM_NUM_PRIORITIES];		 \
    /* 8 */								 \
    gm_port_unprotected_lanai_side_lp_t PORT;				 \
    gm_port_protected_lanai_side_lp_t next_to_poll;			 \
    /* 8 */								 \
    gm_u32_n_t progress_timeout;					 \
    gm_u32_n_t progress_num_lanai_recvs;				 \
    /* 8 */								 \
    gm_yp_query_context_t yp

struct gm_port_protected_lanai_side_fields
{
  GM_PORT_PROTECTED_LANAI_SIDE_FIELDS;
};

/* The protected part of of a user port.  This part of the user port
   state is NOT mapped into user space, and can, therefore, be
   trusted. */

typedef struct gm_port_protected_lanai_side
{
  GM_PORT_PROTECTED_LANAI_SIDE_FIELDS;
#if GM_MIN_SUPPORTED_SRAM > 256
  GM_POW2_PAD (sizeof (struct gm_port_protected_lanai_side_fields));
#endif
}
gm_port_protected_lanai_side_t;

#if GM_MIN_SUPPORTED_SRAM > 256
/* */ GM_TOP_LEVEL_ASSERT (GM_POWER_OF_TWO (sizeof
					    (gm_port_protected_lanai_side_t)));
#endif

/****************
 * event indices.
 ****************/

/* This definition is here only to define NUM_EVENT_TYPES, which is used
   to define the gm_lanai_globals structure below. */

enum gm_event_index
{
  /* Hardware dispatch events in order of increasing priority for LX */
  
  PARITY_INT_EVENT,
  TIMER_EVENT,
  FINISH_RDMA_EVENT,
  FINISH_RECV_PACKET_EVENT,
  FINISH_SDMA_EVENT,
  FINISH_SEND_EVENT,
  START_SEND_EVENT,
  FAIR_SDMA_RDMA_EVENT,
  FAIR_SDMA_EVENT,
  FAIR_RDMA_EVENT,
  POLL_EVENT,

  /* Software dispatch events */

  DMA_PANIC_EVENT,
  DMA_INT_EVENT,
  SDMA_INT_EVENT,
  RDMA_INT_EVENT,
  RECV_BUFFER_OVERFLOW_EVENT,
  START_RECV_PACKET_EVENT,

  /* Events not reached by dispatch, but stored in the handler table. */
  
  START_SDMA_EVENT,
  START_RDMA_EVENT,
  
  /* number of events */
  
  NUM_EVENT_TYPES
};

#include "gm_dma_chain.h"

/**********************************************************************/
/* A structure containing as many GM globals as possible.

   Grouping the globals into a single structure makes them easy to
   access from the host and also allows the compiler to more
   efficiently access the globals under some circumstances. */

typedef struct gm_lanai_globals
{
  gm_u32_n_t trashed_by_event_index_table_HACK;
  gm_u32_n_t magic;
  /* 8 */
  gm_u32_n_t length;
  gm_u32_n_t initialized;
  /* 8 */
  gm_u32_n_t reserved_before_etext;
  gm_lp_n_t etext;
  
  /****************
   * handler table
   ****************/

  /* HACK: HANDLER must be within 256 bytes of the start of this
     struct. */

  gm_handler_n_t handler[NUM_EVENT_TYPES + NUM_EVENT_TYPES % 2];
  /* 8 */

  /****************
   * small fields
   ****************/

  /* Partword fields should appear in the first 2KB of the
     globals for more efficient addressing. */

  gm_u32_n_t this_global_id;	/* packed MAC addr. */
  gm_u8_n_t gm_mac_address[6];    /* unpacked MAC addr. */
  gm_u8_n_t reserved_after_gm_mac_address[6];
  /* 8 */

  /****************
   * Structs with small fields that need to be near the start of this
   * struct (in the first 2K) for efficient addressing.
   ****************/

  struct 
  {
    gm_dma_descriptor_t descriptor[4];
    /* 8 */
    gm_dma_descriptor_lp_t tail_dma_descriptor;
    gm_dma_descriptor_lp_t free_descriptor;
    /* 8 */
    gm_dma_descriptor_lp_t lx_prev_descriptor_filled;
    gm_u32_t pad;
  } sync_dma;
  /* 8 */
  gm_dma_descriptor_t user_dma_descriptor[GM_MAX_NUM_DMA_SEGMENTS+1];
  /* 8 */
  gm_u32_t dummy_flush_target[2];
  /* 8 */

  /****************
   * Frequently used fields
   ****************/

  struct
  {
    gm_port_protected_lanai_side_lp_t port;
    gm_u32_n_t reserved_after_port;
    /* 8 */
    gm_lp_n_t idle_handler;
    gm_lp_n_t active_handler;
  }
  poll;
  /* 8 */

  gm_port_unprotected_lanai_side_lp_t _PORT;
  gm_s32_n_t rand_seed;
  /* 8 */
  /* offset for converting lanai addresses to host addresses. */
  gm_port_protected_lanai_side_lp_t first_port_with_sent_packets;
  gm_subport_lp_t free_subports;
  /* 8 */
				/********************************************
				 * State machine state
				 ********************************************/
  gm_u32_n_t state;
  gm_s32_n_t reserved_after_state;
  /* RECV state */
  /* 8 */
  gm_port_protected_lanai_side_lp_t current_rdma_port;
  gm_port_protected_lanai_side_lp_t registered_raw_recv_port;
  /* 8 */
  gm_u32_n_t reserved_before_first_connection_to_ack;
  gm_connection_lp_t first_connection_to_ack;
  /* 8 */
  gm_recv_token_lp_t recv_token_bin[GM_RECV_TOKEN_HASH_BINS];
  /* 8 */
  gm_connection_lp_t first_active_connection;
  gm_send_record_lp_t free_send_records;
  /* 8 */
  gm_send_record_lp_t first_required_explicit_ack;
  gm_send_record_lp_t last_required_explicit_ack;
  /* 8 */
  /* Buffer for staging recv token DMAs */
  struct _gm_recv_event recv_token_dma_stage;
  /* 8 */
  gm_failed_send_event_t failed_send_event_dma_stage;
  /* 8 */
  struct
  {
    gm_u8_n_t _reserved[GM_RDMA_GRANULARITY - 1];
    gm_u8_n_t type;
  }
  report_dma_stage;
  /* 8 */
  /* the following three variables are made 32 bits for the Alpha,
     which can only perform 32-bit reads and write. */
  struct command_descriptor 
  {
    gm_u32_n_t type;            /* enum gm_lanai_command */
    gm_u32_n_t status;		/* return value */
    union 
    {
      struct
      {
        gm_u32_n_t pad;
        gm_u32_n_t port_num;
      } close;
      struct
      {
        gm_u32_n_t pad;
        gm_u32_n_t port_num;
        /* 8 */
        gm_up_n_t uvma;
        gm_up_n_t len;
      } deregister;
      struct
      {
	gm_u32_n_t global_id;
	gm_u32_n_t node_id;	/* out */
      } get_node_id;
      struct 
      {
        gm_u32_n_t pad;
        gm_u32_n_t port_num;
      } open;
      struct
      {
        gm_u32_n_t pad;
        gm_u32_n_t port_num;
      } uncache_port_ptes;
      struct
      {
	gm_ethernet_mac_addr_t mac;
      } set_ethernet_mac;
      struct
      {
	gm_u8_n_t route[GM_MAX_NETWORK_DIAMETER];
	gm_u8_n_t pad[8 - (GM_MAX_NETWORK_DIAMETER) % 8];
	/* 8 */
	gm_u32_n_t route_len;
	gm_u32_n_t global_id;
      } set_route;
      struct
      {
	gm_u32_n_t port_num;
	gm_u32_n_t reserved;
      } yp;
    } data;
  } command;
  /* 8 */
  gm_port_protected_lanai_side_lp_t finishing_rdma_for_port;
  gm_port_protected_lanai_side_lp_t first_port_with_alarm;
  /* 8 */
  volatile gm_u32_n_t resume_after_halt;
  volatile gm_s32_n_t volatile_zero;	/* BAD */
  /* 8 */
  struct
  {
    /* specify how to create DMA addrs on machines that don't have
       virtual memory. */
    gm_dp_n_t clear;
    gm_dp_n_t set;
  }
  no_VM;
  struct
  {
    gm_lp_n_t pushed_sdma_handler;
    gm_lp_n_t pushed_rdma_handler;
  }
  timer_state;

  struct
  {
    gm_dma_descriptor_t fake_terminal_dma_descriptor;
  } dma;

  struct
  {
    gm_dma_descriptor_lp_t tail_dma_descriptor;
    gm_u32_t reserved_after_tail_dma_descriptor;
  }
  sdma;
  
  struct
  {
    gm_myrinet_packet_descriptor_free_list_t free_list;
    /* 8 */
    gm_myrinet_packet_descriptor_list_t pending;
    /* 8 */
    gm_myrinet_packet_descriptor_lp_t descriptor_being_filled;
    gm_myrinet_packet_segment_lp_t segment_for_continuation;
    /* 8 */
    struct 
    {
      gm_lp_n_t smp;
      gm_lp_n_t smh;
      gm_lp_n_t smlt;
      gm_u32_n_t reserved_after_smlt;
    }  	     
    shortcut;
    /* 8 */
    struct
    {  
      gm_myrinet_packet_descriptor_list_t in_progress;
    } lx;
  }    
  send;

  struct
  {
    gm_myrinet_packet_descriptor_free_list_t free_list;
    /* 8 */
    gm_myrinet_packet_descriptor_lp_t descriptor_being_filled;
    gm_u32_t reserved;
    /* 8 */
    struct 
    {
      gm_lx_received_packet_descriptor_lp_t current_rpd;
      gm_lx_extended_receive_buffer_lp_t current_erb;
    } lx;
  }
  recv;

  struct
  {
    gm_myrinet_packet_descriptor_list_t pending;
    /* 8 */
    gm_myrinet_packet_descriptor_lp_t current_descriptor;
    gm_myrinet_packet_segment_lp_t segment_for_continuation;
    /* 8 */
    gm_dma_descriptor_lp_t tail_dma_descriptor;
    gm_lp_n_t reserved_after_tail_descriptor;
    /* 8 */
    gm_myrinet_packet_descriptor_free_list_t get_reply_free_list;
    gm_recv_queue_slot_t recv_queue_stage;
  }
  rdma;

  struct
  {
    gm_myrinet_packet_descriptor_lp_t current_reply;
    gm_port_protected_lanai_side_lp_t current_got_port;
    /* 8 */
    gm_send_token_lp_t pass_to_host_when_done;
    gm_dma_descriptor_lp_t tail_dma_descriptor;
  }
  get;
  
  /****************
   * infrequently used fields
   ****************/

  gm_u8_n_t hostname[GM_MAX_HOST_NAME_LEN];
  gm_u8_n_t zero_after_hostname;
  gm_u8_n_t reserved_after_zero_after_hostname[7];
  
  /* 8 */
  union gm_interrupt_descriptor interrupt;
  /* 8 */
  gm_dp_n_t interrupt_dp;
#if GM_SIZEOF_DP_T == 4
  gm_u32_n_t pad;
#endif
  /* 8 */
  struct 
  {
#if GM_SIZEOF_DP_T == 4
    gm_u32_n_t pad;
#endif
    gm_dp_n_t addr;
    /* 8 */
    gm_u32_n_t enabled;		/* 32-bits for fast access. */
    gm_u16_n_t data;
    gm_u16_n_t reserved_after_data;
  } msi;			/* message signalled interrupt state */
  /* 8 */
  gm_s32_n_t timeout_time;
  gm_s32_n_t sram_length;
  /* 8 */
  gm_s32_n_t led;
  gm_u32_n_t software_loopback_disabled;
  /* 8 */
  gm_u32_n_t l2e_time[4];
  gm_u32_n_t e2l_time[4];
  /* 8 */
  gm_u32_n_t bus_width;
  gm_u32_n_t bus_rate;
  /* 8 */
  gm_u8_n_t trash[8];
  /* 8 */
  gm_u8_n_t eight_zero_bytes[8];
  /* 8 */

  /****************
   * Large ethernet state
   ****************/

  /* The ethernet state should be near the end so that the GM state
     above can be addressed more efficiently using smaller
     offsets from the pointer to this structure. */

  /* 8 */
  struct
  {
    /* 8 */
    gm_u8_t mac_address[6];
    gm_u8_t reserved_after_mac_address;
    gm_u8_t tag;
    /* 8 */
    struct
    {
      /* 8 */
      gm_send_token_lp_t send_token;
      gm_myrinet_packet_descriptor_lp_t mpd;
      /* 8 */
      gm_port_protected_lanai_side_lp_t port;
      gm_u32_t reserved;
      /* 8 */
      gm_ethernet_gather_list_t gather_list;
      /* 8 */
      gm_u8_n_t assign_serial_number;
      gm_u8_n_t report_serial_number;
      gm_u8_n_t reserved_after_report_serial_number[6];
      /* 8 */
      gm_send_token_lp_t early_completion
          [GM_SLOW_POW2_ROUNDUP(GM_NUM_SEND_TOKENS)];
      /* 8 */      
    }
    sdma;
    /* 8 */
    struct
    {
      gm_ethernet_scatter_list_t scatter_list;
      /* 8 */
      gm_up_n_t scatter_list_ptr;
      gm_dma_descriptor_lp_t first_checksum_descriptor;
      gm_dma_descriptor_lp_t last_checksum_descriptor_next;
#if GM_SIZEOF_UP_T == 4
      gm_u32_t reserved_after_last_checksum_descriptor_next;
#endif
      /* 8 */
      gm_myrinet_packet_descriptor_lp_t mpd;
      gm_u16_n_t checksum;
      gm_u16_n_t total_len;
    }
    rdma;
    struct
    {
      gm_u8_n_t	credits;
      gm_u8_n_t	max_credits;
      gm_u8_n_t	credit_incr;
      gm_u8_n_t	deferred;
      gm_u16_n_t sdmas_until_wake;
      gm_u16_n_t max_sdmas_until_wake;
    }
    wake;
    /* 8 */
    gm_u8_n_t addr_stage[8];
    /* 8 */
    gm_dp_n_t addr_table_piece[GM_MAX_NUM_ADDR_TABLE_PIECES];
    /* 8 */
    gm_u32_n_t mtu;
    gm_u32_n_t flags;
  }
  ethernet;
  /* 8 */

  /****************
   * Mapper state
   ****************/
  struct {
    gm_lanai_mapper_state_t mapper_state;
    /* 8 */
    gm_myrinet_packet_descriptor_free_list_t scout_reply_free_list;
    /* 8 */
  }
  mapper;
  /* 8 */


  /****************
   * Debug stuff
   ****************/

  /* Debug stuff should be close to the end since indexing to
     large offsets is more costly, and when the debug code is not
     used, we would like to save the small offsets for more
     important stuff. */

  gm_u8_lp_n_t event_index_table;
  gm_u32_n_t reserved_after_event_index_table;
  /* 8 */
  gm_lp_n_t dispatch_seen[GM_ROUNDUP (u32, GM_DISPATCH_MAX_NUM, 2)];
  /* 8 */
  gm_u32_n_t dispatch_cnt[GM_ROUNDUP (u32, GM_DISPATCH_MAX_NUM, 2)];
  /* 8 */
  gm_u32_n_t pause_cnt;
  gm_u32_n_t logtime_index;
  /* 8 */
  gm_s32_n_t record_log;
  gm_firmware_log_entry_lp_t log_slot;
  /* 8 */
  gm_firmware_log_entry_lp_t log_end;
  gm_s8_lp_n_t lzero;
  /* 8 */
  gm_firmware_log_entry_t log[GM_LOG_LEN];
  /* 8 */
  volatile gm_s8_lp_n_t current_handler;
  volatile gm_s8_lp_n_t current_handler_extra;
  /* 8 */
  /* debugging info */
  struct
  {
    gm_u32_n_t first_dma_page;
#if GM_SIZEOF_DP_T == 8
    gm_u32_n_t reserved;
#endif
    gm_dp_n_t min_dma_addr;
    /* 8 */
    gm_lp_t old_fp;
    gm_u32_t reserved_after_old_fp;
    /* 8 */
    gm_dma_page_bitmap_t dma_page_bitmap;
  }
  debug;
  /* 8 */
#if GM_ENABLE_TRACE
  gm_u32_n_t trace_active;
  gm_u32_n_t trace_index;
  /* 8 */
  gm_l_trace_t trace_log[GM_LANAI_NUMTRACE];
#endif

  /* Error counters */

#define GM_ERROR_CNT(name, desc) gm_u32_n_t name ## _error_cnt;
  GM_ERROR_COUNTERS
#undef GM_ERROR_CNT
  /* debug counters */
#define GM_DEBUG_CNT(name, desc) gm_u32_n_t name ## _debug_cnt;
  GM_DEBUG_COUNTERS
#undef GM_DEBUG_CNT
  /* counter padding */
  gm_u32_n_t counter_pad[2 - (GM_NUM_ERROR_COUNTERS
			      + GM_NUM_DEBUG_COUNTERS) % 2];
  /* 8 */
  gm_u32_n_t netrecv_cnt;
  gm_u32_n_t netsend_cnt;
  /* 8 */
  gm_sexno_t packet_sexno;
  gm_sexno_t ack_sexno;
  /* 8 */
  gm_u32_n_t while_waiting;
  gm_u32_n_t reserved_after_while_waiting;
  /* 8 */

  /****************
   * Arrays: Large arrays go at the end to reduce the cost of
   * referencing the smaller fields above
   ****************/

  /* Arrays used to allocate storage, but never indexed into except
     during initialization. */

  gm_send_record_t _send_record[GM_NUM_SEND_RECORDS];
  /* 8 */
  gm_subport_t _subport[GM_NUM_SUBPORTS];
  /* 8 */

  /****************
   * big arrays
   ****************/

  /* 8 */
  struct
  {
    /* Description of host-resident page hash table, which is broken
       into pieces. */
    gm_dp_n_t bogus_sdma_ptr;
    gm_dp_n_t bogus_rdma_ptr;
    /* 8 */
    /* A cache of the most recently used hash table entries. */
    gm_page_hash_cache_t cache;
    /* 8 */
  }
  page_hash;

  /* In the MCP, this should be referenced only via the "gm_port" macro
     for best performance. */

  gm_port_protected_lanai_side_t port[GM_NUM_PORTS];
  /* 8 */
  gm_s8_n_t malloc_ram[1 << 18];
  /* 8 */
  gm_u8_lp_n_t last_dispatch;
  gm_s32_n_t last_reported_second;
  /* 8 */
  
  /****************
   * End magic and connections
   ****************/

  gm_u32_n_t event_index_table_crc;
  gm_u32_n_t end_magic;		/* just before connection[] */
  /* 8 */
  /* Connections grow off the end of the globals, consuming as much
     memory as is available, so this should be at the end. */
  struct 
  {
    gm_u32_n_t max_id;
    gm_u32_n_t first_free_id;
    /* 8 */
    gm_u32_n_t bin_id_limit;
    gm_u32_n_t bin_id_ballpark;
    /* 8 */
    /* OPT: Should separate the hash table and connection array for
       faster addressing. */
    gm_connection_t array[1];
  } connection;
}
gm_lanai_globals_t;

/****************
 * The data stored at SRAM offset 0.
 ****************/

typedef struct gm_first_sram_page
{
  gm_u32_n_t init_progress;	/* reflects progress of mcp/gm_initialize() */
  gm_lp_n_t globals;		/* location of the LANai globals */
  /* 8 */
  union
  {
    gm_u64_n_t whole;
    struct 
    {
      gm_u32_n_t low;
      gm_u32_n_t high;
    } parts;
  } rtc;			/* timer-updated extended real time clock */
  /* 8 */
  gm_lp_n_t scratch;
  gm_u32_n_t mcp_status;
} gm_first_sram_page_t;

#if GM_BUILDING_FIRMWARE
#define GM_FIRST_SRAM_PAGE (*(gm_first_sram_page_t *)0)
#endif /* GM_BUILDING_FIRMWARE */

/***********************************************************************
 * Inline functions (used by LANai and host)
 ***********************************************************************/

#define GM_MAX_LENGTH_FOR_SIZE(size) ((gm_u32_t)((size)<3?0:(1<<(size))-8))

#endif /* ifdef _gm_types_h_ */


/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
