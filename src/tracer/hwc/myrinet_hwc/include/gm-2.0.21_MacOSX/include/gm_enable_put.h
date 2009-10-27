/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1998 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_enable_put_h_
#define _gm_enable_put_h_

#define GM_ENABLE_PUT 1
#define GM_DEBUG_PUT 0

/* Disable dirsend for compact builds */

#include "gm_config.h"
#if GM_MIN_SUPPORTED_SRAM <= 256
#undef GM_ENABLE_PUT
#define GM_ENABLE_PUT 0
#endif

#endif /* _gm_enable_put_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
