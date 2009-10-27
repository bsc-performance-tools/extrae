/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: loic@myri.com */

#ifndef _gm_enable_trace_h_
#define _gm_enable_trace_h_

/* WARNING: This feature has not been used in quite a while, and may not
   have been properly maintained.  Use with caution. --Glenn */
#define GM_ENABLE_TRACE 0

/* disable tracing in non-linux kernels, since only Linux supports this
   feature in the kernel. */

#include "gm_config.h"
#if GM_KERNEL && !GM_OS_LINUX && GM_ENABLE_TRACE
#error GM_ENABLE_TRACE is only supported for Linux
#endif

#define GM_HOST_NUMTRACE 4096
#define GM_LANAI_NUMTRACE 2048

#endif /* _gm_enable_trace_h_ */
