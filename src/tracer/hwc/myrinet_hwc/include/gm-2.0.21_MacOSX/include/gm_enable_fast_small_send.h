/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_enable_fast_small_send_h_
#define _gm_enable_fast_small_send_h_

#include "gm_config.h"

#undef  GM_FAST_SMALL_SEND
#define GM_FAST_SMALL_SEND 0

typedef void *gm_ptr_t;
#define GM_FAST_SEND_LEN 32

#endif /*_gm_enable_fast_small_send_h_*/

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
