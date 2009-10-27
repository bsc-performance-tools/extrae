/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2002 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_min_max_h_
#define _gm_min_max_h_

/***********************************************************************
 * min/max functions
 ***********************************************************************/

#include "gm.h"

/****************
 * gm_s32_t
 ****************/

static inline gm_s32_t
gm_s32_min (gm_s32_t a, gm_s32_t b)
{
  return a < b ? a : b;
}

static inline gm_s32_t
gm_s32_max (gm_s32_t a, gm_s32_t b)
{
  return a > b ? a : b;
}

#endif /* _gm_min_max_h_ */
