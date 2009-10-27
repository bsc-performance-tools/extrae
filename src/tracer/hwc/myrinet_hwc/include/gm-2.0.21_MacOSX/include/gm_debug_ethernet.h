/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2001 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_debug_ethernet_h_
#define _gm_debug_ethernet_h_

/* GM_DEBUG_ETHERNET is not in gm_enable_ethernet.h because that file
   is included by gm_types.h, and we don't want to have to recompile
   all of GM just because we turn on ethernet debugging. */

#define GM_DEBUG_ETHERNET 0
#define GM_DEBUG_ETHERNET_BROADCAST 0
#define GM_DEBUG_ETHERNET_RECV_DESCRIPTORS 0

#endif /* _gm_debug_ethernet_h_ */


/*
  This file uses GM standard indentation.

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
