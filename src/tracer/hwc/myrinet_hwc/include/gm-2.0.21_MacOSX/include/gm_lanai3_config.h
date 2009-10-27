/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

#ifndef _gm_lanai3_config_h_
#define _gm_lanai3_config_h_

/*************************************************************************
 * GM native lanai3 configuration.
 * 
 * For Unix, Gnu autoconf scripts are used to generate gm_config.h. 
 * For the lanai3, this is not possible, so we use this file instead.
 *************************************************************************/

#error This file is not used or maintained.

#define GM_SIZEOF_VOID_P 4

#define GM_SIZEOF_SHORT 2
#define GM_SIZEOF_CHAR 1

#undef STDC_HEADERS 1
#undef HAVE_LIMITS_H 1
#undef HAVE_TIME_H 1
#undef HAVE_ERRNO_H
#undef HAVE_FCNTL_H
#undef HAVE_GETPAGESIZE
#undef HAVE_LIBGCC
#undef HAVE_MMAN_H
#undef HAVE_NETINET_IN_H
#undef HAVE_PWD_H
#undef HAVE_STRINGS_H
#undef HAVE_SYS_FILE_H
#undef HAVE_SYS_STAT_H
#undef HAVE_SYS_TYPES_H
#undef HAVE_UNISTD_H

#endif /* _gm_lanai3_config_h_ */
