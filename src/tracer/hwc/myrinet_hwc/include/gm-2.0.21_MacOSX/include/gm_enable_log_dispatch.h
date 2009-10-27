/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2000 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_enable_log_dispatch_h_
#define _gm_enable_log_dispatch_h_

/* Enable MCP logging.  The name "LOG_DISPATCH" is not the most obvious,
   but it makes the mcp/mark_dispatches script do the right thing. */
#define GM_ENABLE_LOG_DISPATCH 0

/* Temporary HACK */
#define GM_LOG_DISPATCHES GM_ENABLE_LOG_DISPATCH

/* Flag to cause LOG_DISPATCH() to also print the log messages.  This
   makes for very slow execution, but is sometimes a good way to get
   more useful information to the debug console so that one can know
   the relationship between the dispatch sequence and the other debug
   information on the debug console. */
#define GM_ENABLE_LOG_DISPATCH_PRINT 0

#endif /* _gm_enable_log_dispatch_h_ */
