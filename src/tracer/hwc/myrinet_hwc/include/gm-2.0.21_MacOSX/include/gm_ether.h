/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

#ifndef _gm_ether_h_
#define _gm_ether_h_

/****************************************************************
 * Includes
 ****************************************************************/

#include "gm_config.h"
#include "gm_cpp.h"
#include "gm_simple_types.h"
#include "gm_sizes.h"

/****************************************************************
 * Constants
 ****************************************************************/

#define GM_ETHERNET_MTU (9*1024)
#define GM_ETHERNET_OVER_MYRINET_MTU					\
(sizeof (struct gm_ethernet_over_myrinet_header) + GM_ETHERNET_MTU)
#define GM_MAX_ETHERNET_WAKE 16

/****************
 * Ethernet types
 ****************/

/* Ethernet MAC address. */

typedef gm_u8_n_t gm_ethernet_mac_addr_t[6];

/* Ethernet packet header.  NOT an multiple of 8 bytes long. */

struct gm_ethernet_header
{
  gm_ethernet_mac_addr_t dest;
  gm_ethernet_mac_addr_t src;
  gm_u16_t type_length;
  /* ethernet payload follows */
};

/* Structure prepended to Ethernet packets when creating
   Ethernet-over-Myrinet packets.  This is NOT a multiple of 8 bytes
   long, so that both the Ethernet and IP headers in
   IP-over-Ethernet-over-Myrinet packets will be naturally aligned. */

struct gm_ethernet_over_myrinet_header
{
  gm_u16_n_t type;
  gm_u16_n_t subtype;
  /** Bytes of ethernet data that follow the prefix. */
  gm_u16_n_t payload_len;
  gm_s16_n_t seqno;
  /** The gateway that injected this packet onto the Myrinet */
  gm_ethernet_mac_addr_t gateway_mac_addr;
  /** payload follows, and is an Ethernet packet. */
};

#define GM_ETHERNET_SUBTYPE_DEFAULT 0
/* Special subtype to request a gateway reply  */
#define GM_ETHERNET_SUBTYPE_UNKNOWN_GATEWAY 1

/*
 * Ethernet flag bits for the MCP
 */
#define GM_ETHER_PROMISC	0x00000001	/* enable promiscuous mode */

/************************************************************************
 * Ethernet types
 ************************************************************************/

typedef struct gm_ethernet_segment_descriptor
{
  gm_dp_t ptr;
  gm_u32_t len;
#if GM_SIZEOF_DP_T == 8
  gm_u32_t pad;
#endif
} gm_ethernet_segment_descriptor_t;

typedef struct gm_ethernet_segment_descriptor_n
{
  gm_dp_n_t ptr;
  gm_u32_n_t len;
#if GM_SIZEOF_DP_T == 8
  gm_u32_n_t pad;
#endif
} gm_ethernet_segment_descriptor_n_t;

#define GM_MAX_ETHERNET_SCATTER_CNT  5

#define GM_MAX_ETHERNET_GATHER_CNT GM_MAX_ETHERNET_SCATTER_CNT

typedef gm_ethernet_segment_descriptor_t
gm_ethernet_scatter_list_t[GM_MAX_ETHERNET_SCATTER_CNT];

typedef gm_ethernet_segment_descriptor_n_t
gm_ethernet_scatter_list_n_t[GM_MAX_ETHERNET_SCATTER_CNT];

typedef gm_ethernet_segment_descriptor_t
gm_ethernet_gather_list_t[GM_MAX_ETHERNET_GATHER_CNT];

typedef gm_ethernet_segment_descriptor_n_t
gm_ethernet_gather_list_n_t[GM_MAX_ETHERNET_GATHER_CNT];

GM_TOP_LEVEL_ASSERT (sizeof (gm_ethernet_scatter_list_t) % 8 == 0);

/****************************************************************
 * funtions prototypes
 ****************************************************************/

void gm_disable_ethernet (struct gm_port *p);

gm_status_t gm_enable_ethernet (struct gm_port *p, 
                                int recv_ring_entries,
                                gm_u16_t tx_intr_freq,
				gm_u32_t mtu);

struct gm_instance_state;
void gm_set_ethernet_flags (struct gm_instance_state *is, unsigned int flags);
void gm_set_ethernet_mtu (struct gm_instance_state *is, gm_u32_t mtu);
gm_u32_t gm_get_ethernet_mtu (struct gm_instance_state *is);
gm_status_t gm_set_ethernet_mac_address (struct gm_instance_state *is,
					 gm_u8_t *addr);

GM_ENTRY_POINT gm_status_t gm_dma_addr (struct gm_port *p,
					gm_up_t ptr,
					gm_dp_t *ret);

GM_ENTRY_POINT void
gm_ethernet_provide_scatter_list_with_tag (struct gm_port *p, 
                                           gm_u32_t segment_cnt,
				           gm_ethernet_segment_descriptor_t
				           segment[],
                                           gm_u8_t tag);

void
gm_ethernet_broadcast (struct gm_port *p,
		       gm_u32_t segment_cnt,
		       gm_ethernet_segment_descriptor_t segment[]);

GM_ENTRY_POINT
void
gm_ethernet_broadcast_with_callback (struct gm_port *p,
				     gm_u32_t segment_cnt,
				     gm_ethernet_segment_descriptor_t
				     /* */ segment[],
				     gm_send_completion_callback_t callback,
				     void *context);

void
gm_ethernet_send (struct gm_port *p, gm_u32_t segment_cnt,
		  gm_ethernet_segment_descriptor_t segment[],
		  gm_u8_t *ethernet_addr);

GM_ENTRY_POINT
void
gm_ethernet_send_with_callback (struct gm_port *p,
				gm_u32_t segment_cnt,
				gm_ethernet_segment_descriptor_t segment[],
				gm_u8_t *ethernet_addr,
				gm_send_completion_callback_t callback,
				void *context);

void
gm_ethernet_set_recv_intr_callback (struct gm_port *p,
				    void (*callback)(void *, 
					             gm_ethernet_recv_t *),
				    void *context);

void
gm_ethernet_set_sent_intr_callback (struct gm_port *p,
				    void (*callback)(void *),
				    void *context);

#endif /* _gm_ether_h_ */
