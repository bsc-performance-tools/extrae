/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* This file provides accessor functions for the packed fields in the
   gm_pte_t type. */

#ifndef _gm_pte_h_
#define _gm_pte_h_

#include "gm_debug.h"
#include "gm_types.h"

#define GM_DEBUG_PTES 0

/****************
 * PTE accessor functions
 ****************/

/* page_port */

static gm_inline gm_up_t
gm_pte_get_page_port (gm_pte_t * e)
{
  return gm_ntoh_up (e->page_port);
}

static gm_inline void
gm_pte_set_page_port (gm_pte_t * e, gm_up_t page_port)
{
  e->page_port = gm_hton_up (page_port);
}

/* packed DMA addr */

static gm_inline gm_dp_t
gm_pte_get_packed_dma_addr (gm_pte_t * e)
{
  return gm_ntoh_dp (e->_packed_dma_addr);
}

static gm_inline void
gm_pte_set_packed_dma_addr (gm_pte_t * e, gm_dp_t packed_dma_addr)
{
  e->_packed_dma_addr = gm_hton_dp (packed_dma_addr);
}

/* user page */

static gm_inline gm_up_t
gm_pte_get_user_page (gm_pte_t * e)
{
  return GM_PAGE_ALIGN (up, gm_pte_get_page_port (e));
}

static gm_inline unsigned int
gm_pte_get_port_id (gm_pte_t * e)
{
  return (unsigned int) (GM_PAGE_OFFSET (gm_pte_get_page_port (e)));
}

/* DMA page */

static gm_inline gm_dp_t
gm_pte_get_dma_page (gm_pte_t * e)
{
  return GM_PAGE_ALIGN (dp, gm_pte_get_packed_dma_addr (e));
}

static gm_inline unsigned int
gm_pte_get_ref_cnt (gm_pte_t * e)
{
  return (unsigned int) (GM_PAGE_OFFSET (gm_pte_get_packed_dma_addr (e)));
}

static gm_inline void
gm_pte_set_dma_page (gm_pte_t * e, gm_dp_t page_addr)
{
  gm_assert (GM_PAGE_ALIGNED (page_addr));
  gm_pte_set_packed_dma_addr (e, page_addr | gm_pte_get_ref_cnt (e));
}

/* reference count */

static gm_inline void
gm_pte_set_ref_cnt (gm_pte_t * e, unsigned int ref_cnt)
{
  gm_assert (GM_PAGE_OFFSET (ref_cnt) == ref_cnt);
  gm_pte_set_packed_dma_addr (e, gm_pte_get_dma_page (e) | ref_cnt);
}

static gm_inline gm_status_t
gm_pte_incr_ref_cnt (gm_pte_t * e)
{
  unsigned int ref_cnt;

  ref_cnt = gm_pte_get_ref_cnt (e);
  if (ref_cnt == GM_PAGE_LEN - 1)
    return GM_PTE_REF_CNT_OVERFLOW;
  gm_pte_set_ref_cnt (e, ref_cnt + 1);
  return GM_SUCCESS;
}

static gm_inline unsigned int
gm_pte_decr_ref_cnt (gm_pte_t * e)
{
  unsigned int ref_cnt;

  ref_cnt = gm_pte_get_ref_cnt (e);
  gm_assert (ref_cnt != 0);
  gm_pte_set_ref_cnt (e, ref_cnt - 1);
  return ref_cnt - 1;
}

/* clearing */

static gm_inline void
gm_pte_clear (gm_pte_t * e)
{
  gm_pte_set_page_port (e, 0);
  /* Leave the DMA address, so in case the firmware is DMAing the
     descriptor during the update, it will always get a valid DMA
     address. */
}

static gm_inline void
gm_pte_print (char *str, gm_pte_t * pte)
{
  GM_PARAMETER_MAY_BE_UNUSED (str);
  GM_PARAMETER_MAY_BE_UNUSED (pte);

  GM_CALLED ();
  
  GM_PRINT (GM_DEBUG_PTES,
	    ("%s:\n"
	     "\tpage=0x%qx\n"
	     "\tport=0x%x\n"
	     "\tdma_addr=0x%qx\n"
	     "\tref_cnt=0x%x\n",
	     str ? str : 0,
	     (gm_u64_t) gm_pte_get_user_page (pte),
	     gm_pte_get_port_id (pte),
	     (gm_u64_t) gm_pte_get_dma_page (pte), gm_pte_get_ref_cnt (pte)));

  GM_RETURN_NOTHING ();
}

#endif /* _gm_pte_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
