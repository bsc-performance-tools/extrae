/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2000 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* This file implements simple macros to support bitmasks.  The loads
   and stores used are 32-bit for maximum performance on most
   architectures, but more critically to avoid partword PIO, which
   some architectures (CPSI MAP26xx boards) do no support.  These
   32-bit words are always accessed in Big Endian fashion, so there is
   no need to worry about Endianness issues when sharing these bitmaps
   in heterogeneous Endian environments. */

#ifndef _gm_bitmap_h_
#define _gm_bitmap_h_

#include "gm.h"

/* special type for bitmap word array, to catch direct bitmap access. */

typedef struct
{
  gm_u32_t __bitmap_u32;
}
gm_bitmap_u32_t;

/* Declare a bitmap containing LEN bits.  Length is rounded up to an
   8-byte boundary. */

#define GM_BITMAP_DECL(name, len) gm_bitmap_u32_t (name)[((len)+63) >> 6 << 1]

/****************
 * Endian-dependent versions
 ****************/

/* set a bit in a bitmap */

#define _GM_BITMAP_SET(bitmap, pos) do {				\
  gm_assert ((unsigned long) (pos) < 8 * sizeof (bitmap));		\
  (bitmap)[(pos) >> 5].__bitmap_u32 |= 1 << ((pos) & 31);		\
} while (0)

/* clear a bit in a bitmap */

#define _GM_BITMAP_CLEAR(bitmap, pos) do {				\
  gm_assert ((unsigned long) (pos) < 8 * sizeof (bitmap));		\
  (bitmap)[(pos) >> 5].__bitmap_u32 &= ~(1 << ((pos) & 31));		\
} while (0)

/* get a bit in a bitmap */

#define _GM_BITMAP_GET(bitmap, pos) ((bitmap)[(pos) >> 5].__bitmap_u32	\
				    & 1 << ((pos) & 31))
/****************
 * Endian-independent versions
 ****************/

/* set a bit in a bitmap that is set in network byte order. */

#if GM_CPU_BIGENDIAN || GM_MCP
#define GM_BITMAP_SET(bitmap, pos) _GM_BITMAP_SET (bitmap, pos)
#else
#define GM_BITMAP_SET(bitmap, pos) _GM_BITMAP_SET (bitmap, (pos) ^ 0x18)
#endif

/* clear a bit in a bitmap that is set in network byte order. */

#if GM_CPU_BIGENDIAN || GM_MCP
#define GM_BITMAP_CLEAR(bitmap, pos) _GM_BITMAP_CLEAR (bitmap, pos)
#else
#define GM_BITMAP_CLEAR(bitmap, pos) _GM_BITMAP_CLEAR (bitmap, (pos) ^ 0x18)
#endif

/* get a bit in a bitmap that is set in network byte order. */

#if GM_CPU_BIGENDIAN || GM_MCP
#define GM_BITMAP_GET(bitmap, pos) _GM_BITMAP_GET (bitmap, pos)
#else
#define GM_BITMAP_GET(bitmap, pos) _GM_BITMAP_GET (bitmap, (pos) ^ 0x18)
#endif

/****************
 * Wrapping versions
 *
 * These versions wrap overflow positions back into the array.  For
 * efficient operation, sizeof (bitmap) should be a power of 2.
 ****************/

#define __GM_BITMAP_WRAP(bitmap, pos) ((pos) % (8 * sizeof (bitmap)))

#define GM_BITMAP_WRAPPING_SET(bitmap, pos)				\
     GM_BITMAP_SET (bitmap, __GM_BITMAP_WRAP (bitmap, pos))
#define GM_BITMAP_WRAPPING_CLEAR(bitmap, pos)				\
     GM_BITMAP_CLEAR (bitmap, __GM_BITMAP_WRAP (bitmap, pos))
#define GM_BITMAP_WRAPPING_GET(bitmap, pos) 				\
     GM_BITMAP_GET (bitmap, __GM_BITMAP_WRAP (bitmap, pos))

/****************
 * Bitmap copying
 *
 * Copy a bitmap a word at a time.  This prevents partword PIO problems.
 ****************/

#define GM_BITMAP_COPY(from, to) do {					\
  gm_always_assert (sizeof (from) == sizeof (to));			\
  __gm_bitmap_copy (from, to, sizeof (from) / 4);			\
} while (0)

static gm_inline void
__gm_bitmap_copy (gm_bitmap_u32_t from[],
		  gm_bitmap_u32_t to[], gm_size_t words)
{
  gm_size_t word;

  for (word = 0; word < words; word++)
    {
      to[words] = from[words];
    }
}

#endif /* _gm_bitmap_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
