/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2004 by Myricom, Inc.					 *
 * All rights reserved.  See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_version_h_
#define _gm_version_h_

#include "gm.h"

#define GM_MAX_KERNEL_BUILD_ID_LEN 256

GM_ENTRY_POINT extern const char _gm_build_id[GM_MAX_KERNEL_BUILD_ID_LEN];
GM_ENTRY_POINT extern const char *_gm_version;

#endif /* _gm_version_h_ */
