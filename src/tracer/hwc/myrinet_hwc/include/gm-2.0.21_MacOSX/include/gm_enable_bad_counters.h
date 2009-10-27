/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_enable_bad_counters_h_
#define _gm_enable_bad_counters_h_

#include "gm_config.h"

#define GM_ENABLE_BAD_COUNTERS 1

#if GM_ENABLE_BAD_COUNTERS
#define GM_INCR_P(c) (++(c))
#define GM_DECR_P(c) (--(c))
#else
#define GM_INCR_P(c)
#define GM_DECR_P(c)
#endif

#endif /* _gm_enable_bad_counters_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
