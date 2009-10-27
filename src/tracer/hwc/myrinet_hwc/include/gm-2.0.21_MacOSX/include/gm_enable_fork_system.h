/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/


#ifndef _gm_enable_fork_system_h
#define _gm_enable_fork_system_h

#include "gm_config.h"

/* 'ENABLE' means allow users to have fork or system in their programs */

#undef  GM_ENABLE_FORK_SYSTEM
#define GM_ENABLE_FORK_SYSTEM (GM_OS_LINUX | GM_OS_SOLARIS)

#endif /* _gm_enable_fork_system_h */
