/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_arch_io_h_
#define _gm_arch_io_h_

#if defined HAVE_SYS_ERRNO_H && !GM_KERNEL
#include <sys/errno.h>
#endif

#if GM_KERNEL
/* These files may not be included in the kernel */
#undef HAVE_SYS_TYPES_H
#undef HAVE_NETINET_IN_H
#endif

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  c-backslash-column:72
  End:
*/

#endif /* _gm_arch_io_h_ */
