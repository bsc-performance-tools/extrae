/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2002 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_constants_h_
#define _gm_constants_h_

/* max length of a string returned by _gm_get_firmware_string().
   Actual firware strings may be longer. */
#define GM_MAX_FIRMWARE_STRING_LEN 128

/* The maximum number of switches that must be traversed to reach any
   remote host.  Multiple of 4. */
#define GM_MAX_NETWORK_DIAMETER 24

/* The number of ticks in a second, approximated to a power of 2. */

#define GM_TICKS_PER_USEC (2)
#define GM_TICKS_PER_MSEC (1024 * GM_TICKS_PER_USEC) /* approximate */
#define GM_TICKS_PER_SEC (1024 * GM_TICKS_PER_MSEC) /* approximate */

#endif /* _gm_constants_h_ */
