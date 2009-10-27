/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1998 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_enable_security_h_
#define _gm_enable_security_h_

/************************************************************
 *
 * EXPLANATION
 *
 * When GM_ENABLE_SECURITY is enabled, only root can run the
 * mapper (to prevent users from trashing the network routing
 * tables) and GM's introspective programs (gm_counters,
 * gm_debug) can only be run by root.
 *
 * We have been delivering GM with GM_ENABLE_SECURITY
 * disabled since approximately GM-1.0, because we deemed
 * these restrictions to impose an undue burden on tech
 * support.
 *
 * One possible enhancement would be to move this flag to
 * gm_auto_config.h, creating a configure flag to toggle it.
 * This would make it easier for us and our customers to
 * build a security-enabled GM.
 *
 * Note that, since we have not enabled this feature in some
 * time, some testing would definitely be in order the next
 * time it is turned on.
 * 
 ***********************************************************/

#define GM_ENABLE_SECURITY 1
#define GM_DEBUG_SECURITY 0

#endif /* _gm_enable_security_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
