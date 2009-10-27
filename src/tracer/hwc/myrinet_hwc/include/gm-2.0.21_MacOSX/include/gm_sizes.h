/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2001 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_sizes_h_
#define _gm_sizes_h_

enum gm_size
{
  GM_SIZE_0 = 0, GM_SIZE_1 = 1, GM_SIZE_2 = 2, GM_SIZE_3 = 3,
  GM_SIZE_4 = 4, GM_SIZE_5 = 5, GM_SIZE_6 = 6, GM_SIZE_7 = 7,
  GM_SIZE_8 = 8, GM_SIZE_9 = 9, GM_SIZE_10 = 10, GM_SIZE_11 = 11,
  GM_SIZE_12 = 12, GM_SIZE_13 = 13, GM_SIZE_14 = 14, GM_SIZE_15 = 15,
  GM_SIZE_16 = 16, GM_SIZE_17 = 17, GM_SIZE_18 = 18, GM_SIZE_19 = 19,
  GM_SIZE_20 = 20, GM_SIZE_21 = 21, GM_SIZE_22 = 22, GM_SIZE_23 = 23,
  GM_SIZE_24 = 24, GM_SIZE_25 = 25, GM_SIZE_26 = 26, GM_SIZE_27 = 27,
  GM_SIZE_28 = 28, GM_SIZE_29 = 29, GM_SIZE_30 = 30, GM_SIZE_31 = 31,
  GM_RAW_TAG_SIZE = 32,		/* used to tag raw sends */
  /* 33 */
  GM_ETHERNET_TAG_SIZE = 34,
  GM_NUM_SIZES
};

#endif /* _gm_sizes_h_ */

/*
  This file uses GM standard indentation.

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
