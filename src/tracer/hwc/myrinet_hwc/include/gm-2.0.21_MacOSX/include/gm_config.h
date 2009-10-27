#ifndef _gm_config_h_		/* -*-c-*- */
#define _gm_config_h_

/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation                      *
 * Copyright (c) 1999 by Myricom, Inc.                            	 *
 * All rights reserved.  See the file `COPYING' for copyright notice.    *
 *************************************************************************/

/* Include this first so that configuration can be made a function of
   who's compiling the software.  This is useful for debugging. */

#include "gm_roster.h"

/************
 * Platform-dependent configurations
 ************/

/* Determine how the source tree will be configured.  For lanai3
   embedded applications, custom header files are used.  For all other
   builds, an automatically constructed header file is used. */

#define GM_MAY_INCLUDE_AUTO_CONFIG_H
#include "gm_auto_config.h"
#undef GM_MAY_INCLUDE_AUTO_CONFIG_H

/* infer GM_MIN_PAGE_LEN */

#if GM_PAGE_LEN			/* GM_PAGE_LEN is constant */
#define GM_MIN_PAGE_LEN GM_PAGE_LEN
#elif GM_SUPPORT_0K_PAGES || GM_SUPPORT_4K_PAGES
#define GM_MIN_PAGE_LEN 4096
#elif GM_SUPPORT_8K_PAGES
#define GM_MIN_PAGE_LEN 8192
#elif GM_SUPPORT_16K_PAGES
#define GM_MIN_PAGE_LEN 16384
#elif GM_SUPPORT_64K_PAGES
#define GM_MIN_PAGE_LEN 65536
#else
#error
#endif

/* infer GM_MAX_PAGE_LEN */

#if GM_PAGE_LEN			/* GM_PAGE_LEN is constant */
#define GM_MAX_PAGE_LEN GM_PAGE_LEN
#elif GM_SUPPORT_64K_PAGES
#define GM_MAX_PAGE_LEN 65536
#elif GM_SUPPORT_16K_PAGES
#define GM_MAX_PAGE_LEN 16384
#elif GM_SUPPORT_8K_PAGES
#define GM_MAX_PAGE_LEN 8192
#elif GM_SUPPORT_4K_PAGES | GM_SUPPORT_0K_PAGES
#define GM_MAX_PAGE_LEN 4096
#else
#error
#endif

#endif /* _gm_config_h_ */
