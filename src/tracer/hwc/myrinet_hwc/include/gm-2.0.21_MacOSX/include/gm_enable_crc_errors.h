/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2000 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_enable_crc_errors_h_
#define _gm_enable_crc_errors_h_

/* Setting this simulates CRC errors in received packets */
#define GM_ENABLE_CRC_ERRORS 0

/* 1 in GM_CRC_RATE packets will be marked as CRC error if enabled */
#define GM_CRC_RATE 50

#define GM_DEBUG_CRC_ERRORS 0

#endif /* _gm_enable_crc_errors_h_ */
