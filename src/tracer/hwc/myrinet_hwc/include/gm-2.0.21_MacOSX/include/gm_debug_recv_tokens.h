/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1998 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_debug_recv_tokens_h_
#define _gm_debug_recv_tokens_h_ 1

#define GM_DEBUG_RECV_TOKENS 0

#include "gm.h"			/* for GM_NUM_PRIORITIES */
#include "gm_types.h"		/* for GM_NUM_SIZES */

/* This is a HACK, because it assumes only one port is in use. */
extern unsigned int gm_debug_recv_token_cnt[GM_NUM_PRIORITIES][GM_NUM_SIZES];

#endif /* _gm_debug_recv_tokens_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
