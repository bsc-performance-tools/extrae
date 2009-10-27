/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2003 by Myricom, Inc.					 *
 * All rights reserved.  See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_debug_gateway_cache_h_
#define _gm_debug_gateway_cache_h_

#define GM_DEBUG_GATEWAY_CACHE 0

/* Set this to force the gateway cache to be used even for MAC
   addresses on the local network (not behind gateways).  This allows
   us to test the gateway caching software on a Myrinet-only
   network. */
   
#define GM_DEBUG_GATEWAY_CACHE__ALWAYS_USE 0

#endif /* _gm_debug_gateway_cache_h_ */
