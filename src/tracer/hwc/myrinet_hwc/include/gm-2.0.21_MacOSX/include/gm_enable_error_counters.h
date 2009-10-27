/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* This file enables error counters */

#ifndef _gm_enable_error_counters_h_
#define _gm_enable_error_counters_h_

#define GM_ENABLE_ERROR_COUNTERS 1

/* Disable error counters for compact builds */

#include "gm_config.h"
#if GM_MIN_SUPPORTED_SRAM <= 256
#undef GM_ENABLE_ERROR_COUNTERS
#define GM_ENABLE_ERROR_COUNTERS 0
#endif

/* handy related macros */

#if GM_ENABLE_ERROR_COUNTERS
#define GM_INCR_ERROR_CNT(c) (++(c ## _error_cnt))
#define GM_DECR_ERROR_CNT(c) (--(c ## _error_cnt))
#define GM_READ_ERROR_CNT(c) (c ## _error_cnt)
#else
#define GM_INCR_ERROR_CNT(c)
#define GM_DECR_ERROR_CNT(c)
#define GM_READ_ERROR_CNT(c) 0
#endif

#endif /* _gm_enable_error_counters_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
