/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2000 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_dma_chain_h_
#define _gm_dma_chain_h_

#include "gm.h"
#include "gm_simple_types.h"

#if GM_BUILDING_FIRMWARE
#include "gm_lanai_def.h"
#endif

/****************************************************************
 * Data structures
 ****************************************************************/

/* HACK to define "struct gm_dma_descriptor" differently on the host
   and on each LANai version.  The idea is that the firmware sees only
   the "native" DMA descriptor, but the host sees all definitions in
   a union.  In all cases, the resulting structure MUST BE THE SAME SIZE.
   This keeps lots of vacuous changes out of the firmware, while still
   allowing the host to access the structure. */

#if !GM_BUILDING_FIRMWARE
#define gm_dma_descriptor_all gm_dma_descriptor
#elif LX
#define gm_dma_descriptor_lx gm_dma_descriptor
#else  /* !LX */
#define gm_dma_descriptor_pre_lX gm_dma_descriptor
#endif /* GM_BUILDING_FIRMWARE */

/* The hardware DMA descriptor for pre-LANaiX */

struct gm_dma_descriptor_pre_lX {
  gm_lp_n_t next_with_flags;	/* next pointer */
  gm_u16_n_t csum[2];	/* 2 16-bit ones complement checksums of this block */
  /* 8 */
  gm_u32_n_t len;		/* byte count */
  gm_lp_n_t lanai_addr;		/* lanai address */
  /* 8 */
  gm_u32_n_t ebus_addr_high;
  gm_u32_n_t ebus_addr_low;
};

/* The hardware DMA descriptor layout for the LANaiX */

struct gm_dma_descriptor_lx
{
  gm_u32_n_t dma_length_with_flags; /* + READ,COUNT,FLUSH,NO_SNOOP,NO_CACHE */
  gm_u32_n_t dma_pci_addr;	/* low 32 bits of PCI addr */
  /* 8 */
  gm_lp_n_t dma_lanai_addr;	/* top 4 bits reserved */
  gm_u32_n_t dma_pci64_addr;	/* high 32 bits of PCI addr */
  /* 8 */
  gm_lp_n_t dma_next_with_flags; /* DMA_APPEND + DMA_VALID */
  gm_u16_n_t dma_offset;	/* skip before computing IP checksum */
  gm_u16_n_t dma_checksum;	/* checksum is stored here. */
};

struct gm_dma_descriptor_all
{
  union 
  {
    struct gm_dma_descriptor_lx lx;
    struct gm_dma_descriptor_pre_lX pre_lX;
  } format;
};

typedef struct gm_dma_descriptor gm_dma_descriptor_t;

/****************************************************************
 * flags
 ****************************************************************/

#define GM_LX_DMA_LENGTH_FLAG_READ (1<<31) /* DMA_READ */
#define GM_LX_DMA_LENGTH_FLAG_COUNT (1<<30) /* DMA_COUNT */
#define GM_LX_DMA_LENGTH_FLAG_FLUSH (1<<29) /* DMA_FLUSH */
#define GM_LX_DMA_LENGTH_FLAG_NO_SNOOP (1<<28) /* DMA_NO_SNOOP */
#define GM_LX_DMA_LENGTH_FLAG_NO_CACHE (1<<27) /* DMA_NO_CACHE */
#define GM_LX_DMA_DMA_LENGTH_MASK 0x00ffffff /* DMA_LENGTH */

#define GM_LX_DMA_NEXT_FLAG_APPEND (1<<29) /* DMA_APPEND */
#define GM_LX_DMA_NEXT_FLAG_VALID (1<<28) /* DMA_VALID */
#define GM_LX_DMA_NEXT_MASK 0x0fffffff /* DMA_NEXT */

#if GM_BUILDING_FIRMWARE && 0
#define GM_LX_SDMA_POINTER DMA0_POINTER
#define GM_LX_SDMA_COUNT DMA0_COUNT
#define GM_LX_RDMA_POINTER DMA1_POINTER
#define GM_LX_RDMA_COUNT DMA1_COUNT
#endif

/* Generic flags. */

#define GM_DMA_DESCRIPTOR_L2E		0x0
#define GM_DMA_DESCRIPTOR_E2L		0x1
#define GM_DMA_DESCRIPTOR_DIR		0x1 /* mask to extract direction */
#define GM_DMA_DESCRIPTOR_TERMINAL	0x2 /* terminal block? */
#define GM_DMA_DESCRIPTOR_WAKE		0x4 /* wake when block completes */
#define GM_DMA_DESCRIPTOR_FLAG_MASK	0x7

/****************************************************************
 * Inlines
 ****************************************************************/

#if GM_BUILDING_FIRMWARE
#if !LX

static inline
gm_size_t
GM_DMA_DESCRIPTOR_NEXT_WITH_FLAGS_FLAGS (gm_lp_t next_with_flags)
{
  return ((gm_size_t) next_with_flags);
}

static inline
gm_dma_descriptor_t *
GM_DMA_DESCRIPTOR_NEXT_WITH_FLAGS_PTR (gm_lp_t next_with_flags)
{
  return ((gm_dma_descriptor_t *)
	  ((gm_size_t) next_with_flags
	   & ~(gm_size_t) GM_DMA_DESCRIPTOR_FLAG_MASK));
}

static inline
gm_lp_t
GM_DMA_DESCRIPTOR_BUILD_NEXT_WITH_FLAGS (gm_dma_descriptor_t *next,
					 int flags)
{
  return (gm_lp_t) ((char *) next + flags);
}

#endif /* !LX */
#endif /* GM_MCP */
#endif /* _gm_dma_chain_h_ */
