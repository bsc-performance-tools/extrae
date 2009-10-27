/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2000-2001 by Myricom, Inc.				 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* This file defines simple (non-aggregate) types that were not
   defined in gm.h.  They are not defined there because they may
   depend on gm_config.h, which may not be included in gm.h.  */

#ifndef _gm_simple_types_h_
#define _gm_simple_types_h_

#include "gm.h"
#include "gm_config.h"

/* Macro to allow structure pointers on the LANai to be defined as
   pointers on the LANai, but as gm_u32_n_t's on the host */

#if GM_MCP
#define GM_LP_N_T(foo) typedef foo *
#else
#define GM_LP_N_T(foo) typedef gm_lp_n_t
#endif

#define GM_TYPEDEF_LP_N_T(type, name) GM_LP_N_T (type name) name ## _lp_t

/****************************************************************
 * Simple types not defined in gm.h
 ****************************************************************/

/****************
 * Host byte order types
 ****************/

#if GM_SIZEOF_DP_T == 8
typedef gm_u64_t gm_dp_t;
#elif GM_SIZEOF_DP_T == 4
typedef gm_u32_t gm_dp_t;
#else
#error unsupported GM_SIZEOF_DP_T
#endif

/****************
 * Network byte order types
 ****************/

#if !GM_MCP

#if GM_STRONG_TYPES
typedef gm_u32_t gm_lp_t;
typedef struct
{
  gm_dp_t n;
}
gm_dp_n_t;

typedef struct
{
  gm_lp_t n;
}
gm_lp_n_t;
#else  /* !GM_STRONG_TYPES */
typedef gm_u32_t gm_lp_t;
typedef gm_dp_t gm_dp_n_t;
typedef gm_lp_t gm_lp_n_t;
#endif /* !GM_STRONG_TYPES */

#else /* GM_MCP */

typedef void *gm_lp_t;
typedef gm_dp_t gm_dp_n_t;
typedef gm_lp_t gm_lp_n_t;

#endif /* GM_MCP */

/****************************************************************
 * LANai char pointer
 ****************************************************************/

#if GM_MCP
typedef char *gm_char_lp_t;
typedef const char *gm_const_char_lp_t;
#else
typedef gm_lp_n_t gm_char_lp_t;
typedef gm_lp_n_t gm_const_char_lp_t;
#endif

/****************************************************************
 * Simple endian conversions not in gm.h
 ****************************************************************/

/****
 * Pointer conversion macros:
 * "dp" versions convert DMA pointers (32 bits)
 * "lp" versions convert LANai pointers (32 bits)
 * "up" versions convert user pointers (64 or 32 bits)
 ****/

/****************
 * DMA pointers
 ****************/

gm_inline static gm_dp_n_t
gm_hton_dp (gm_dp_t dp)
{
  gm_dp_n_t ret;

#if GM_SIZEOF_DP_T == 8
  GM_N (ret) = __gm_hton_u64 (dp);
#else
  GM_N (ret) = __gm_hton_u32 (dp);
#endif
  return ret;
}

gm_inline static gm_dp_t
gm_ntoh_dp (gm_dp_n_t dp)
{
#if GM_SIZEOF_DP_T == 8
  return __gm_ntoh_u64 (GM_N (dp));
#elif GM_SIZEOF_DP_T == 4
  return __gm_ntoh_u32 (GM_N (dp));
#endif
}

/****************
 * LANai pointers
 ****************/

gm_inline static gm_lp_n_t
gm_hton_lp (gm_lp_t lp)
{
  gm_lp_n_t ret;

  GM_N (ret) = (gm_lp_t) __gm_hton_u32 ((gm_u32_t) lp);
  return ret;
}

gm_inline static gm_lp_t
gm_ntoh_lp (gm_lp_n_t lp)
{
  return (gm_lp_t) __gm_ntoh_u32 ((gm_u32_t) GM_N (lp));
}

/****************
 * Remote user pointers (assumed to be 64 bits)
 ****************/

gm_inline static gm_remote_ptr_n_t
gm_hton_rp (gm_remote_ptr_t rp)
{
  gm_remote_ptr_n_t ret;

  GM_N (ret) = (gm_remote_ptr_t) __gm_hton_u64 ((gm_u64_t) rp);
  return ret;
}

gm_inline static gm_remote_ptr_t
gm_ntoh_rp (gm_remote_ptr_n_t rp)
{
  return (gm_remote_ptr_t) __gm_ntoh_u64 ((gm_u64_t) GM_N (rp));
}

/****************
 * User pointers
 ****************/

gm_inline static gm_up_n_t
gm_hton_up (gm_up_t _gm_up)
{
  gm_up_n_t ret;

#if GM_SIZEOF_UP_T == 8
  GM_N (ret) = __gm_hton_u64 (_gm_up);
#else
  GM_N (ret) = __gm_hton_u32 (_gm_up);
#endif
  return ret;
}

gm_inline static gm_up_t
gm_ntoh_up (gm_up_n_t _gm_up)
{
#if GM_SIZEOF_UP_T == 8
  return __gm_ntoh_u64 (GM_N (_gm_up));
#elif GM_SIZEOF_UP_T == 4
  return __gm_ntoh_u32 (GM_N (_gm_up));
#endif
}

#endif /* _gm_simple_types_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
