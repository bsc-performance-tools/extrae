/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gcc_version_h_
#define _gcc_version_h_

#define GCC(major,minor) (((major) << 8) | (minor))
#define GCC_VERSION GCC (__GNUC__, __GNUC_MINOR__)

#endif /* _gcc_version_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
