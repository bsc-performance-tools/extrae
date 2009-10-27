/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2002 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* This fine contains useful bit twiddling hacks. */

#ifndef _gm_bit_twiddling_h_
#define _gm_bit_twiddling_h_

/* Determine log2 of "x", rounded up.  If "x" is a constant, this is done
   at compile time.  Otherwise, you probably don't want to use this macro. */

#define GM_SLOW_LOG2_ROUNDUP(x)						\
((x) <= 0x00000001 ?  0 : (x) <= 0x00000002 ?  1 :			\
 (x) <= 0x00000004 ?  2 : (x) <= 0x00000008 ?  3 :			\
 (x) <= 0x00000010 ?  4 : (x) <= 0x00000020 ?  5 :			\
 (x) <= 0x00000040 ?  6 : (x) <= 0x00000080 ?  7 :			\
 (x) <= 0x00000100 ?  8 : (x) <= 0x00000200 ?  9 :			\
 (x) <= 0x00000400 ? 10 : (x) <= 0x00000800 ? 11 :			\
 (x) <= 0x00001000 ? 12 : (x) <= 0x00002000 ? 13 :			\
 (x) <= 0x00004000 ? 14 : (x) <= 0x00008000 ? 15 :			\
 (x) <= 0x00010000 ? 16 : (x) <= 0x00020000 ? 17 :			\
 (x) <= 0x00040000 ? 18 : (x) <= 0x00080000 ? 19 :			\
 (x) <= 0x00100000 ? 20 : (x) <= 0x00200000 ? 21 :			\
 (x) <= 0x00400000 ? 22 : (x) <= 0x00800000 ? 23 :			\
 (x) <= 0x01000000 ? 24 : (x) <= 0x02000000 ? 25 :			\
 (x) <= 0x04000000 ? 26 : (x) <= 0x08000000 ? 27 :			\
 (x) <= 0x10000000 ? 28 : (x) <= 0x20000000 ? 29 :			\
 (x) <= 0x40000000 ? 30 : 31)

/* Determine the smallest power of 2 larger than x.  If "x" is a
   constant, this is done at compile time.  Otherwise, you probably
   don't want to use this macro. */
#define GM_SLOW_POW2_ROUNDUP(x) (1<<GM_SLOW_LOG2_ROUNDUP(x))

#endif /* _gm_bit_twiddling_h_ */
