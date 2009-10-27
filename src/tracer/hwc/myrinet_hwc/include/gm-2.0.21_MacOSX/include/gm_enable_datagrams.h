/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_enable_datagrams_h_
#define _gm_enable_datagrams_h_

#define GM_ENABLE_DATAGRAMS 1
#define GM_ENABLE_PIO_DATAGRAMS 0
#define GM_DEBUG_DATAGRAMS 0

/* Disable datagrams for compact builds */

#include "gm_config.h"
#if GM_MIN_SUPPORTED_SRAM <= 256
#undef GM_ENABLE_DATAGRAMS
#define GM_ENABLE_DATAGRAMS 0
#endif

#endif /* _gm_enable_datagrams_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
