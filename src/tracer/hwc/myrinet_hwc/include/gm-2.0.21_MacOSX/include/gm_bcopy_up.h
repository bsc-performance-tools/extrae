/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2000 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* This file supplies a inlined bcopy implementation for
   _gm_bcopy_up() and gm_bcopy(). */

#ifdef GM_BCOPY3
#error Need to fix this.
#endif

#include "gm.h"

#if GM_BUILDING_FIRMWARE
#include "gm_bootstrap.h"
#include "gm_lanai_def.h"
#endif

static /*  gm_inline */
void
__gm_bcopy_up (const void *from, void *to, gm_size_t len)
{
  const char *f;
  char *t, *limit;
  char a, b, c;

  f = from;
  t = to;
  limit = t + len;

  /* Copy just enough bytes so that 16*N bytes remain to be copied */

  switch (len & 15)
    {
    case 13:
      a = *f++;
      b = *f++;
      c = *f++;
      *t++ = a;
      *t++ = b;
      *t++ = c;
    case 10:
      a = *f++;
      b = *f++;
      c = *f++;
      *t++ = a;
      *t++ = b;
      *t++ = c;
    case 7:
      a = *f++;
      b = *f++;
      c = *f++;
      *t++ = a;
      *t++ = b;
      *t++ = c;
    case 4:
      a = *f++;
      b = *f++;
      c = *f++;
      *t++ = a;
      *t++ = b;
      *t++ = c;
    case 1:
      a = *f++;
      *t++ = a;
      break;
      
    case 14:
      a = *f++;
      b = *f++;
      c = *f++;
      *t++ = a;
      *t++ = b;
      *t++ = c;
    case 11:
      a = *f++;
      b = *f++;
      c = *f++;
      *t++ = a;
      *t++ = b;
      *t++ = c;
    case 8:
      a = *f++;
      b = *f++;
      c = *f++;
      *t++ = a;
      *t++ = b;
      *t++ = c;
    case 5:
      a = *f++;
      b = *f++;
      c = *f++;
      *t++ = a;
      *t++ = b;
      *t++ = c;
    case 2:
      a = *f++;
      b = *f++;
      *t++ = a;
      *t++ = b;
      break;
      
    case 15:
      a = *f++;
      b = *f++;
      c = *f++;
      *t++ = a;
      *t++ = b;
      *t++ = c;
    case 12:
      a = *f++;
      b = *f++;
      c = *f++;
      *t++ = a;
      *t++ = b;
      *t++ = c;
    case 9:
      a = *f++;
      b = *f++;
      c = *f++;
      *t++ = a;
      *t++ = b;
      *t++ = c;
    case 6:
      a = *f++;
      b = *f++;
      c = *f++;
      *t++ = a;
      *t++ = b;
      *t++ = c;
    case 3:
      a = *f++;
      b = *f++;
      c = *f++;
      *t++ = a;
      *t++ = b;
      *t++ = c;
    case 0:
      break;
    }

  /* copy the rest of the bytes */

  gm_assert (t <= limit);
  gm_assert (((limit - t) & 15) == 0);
  while (t < limit)
    {
      /* copy 16 bytes... fast! */
      /*     */ a = *f++;
      /*     */ b = *f++;
      /*     */ c = *f++;
      *t++ = a; a = *f++;
      *t++ = b; b = *f++;
      *t++ = c; c = *f++;
      *t++ = a; a = *f++;
      *t++ = b; b = *f++;
      *t++ = c; c = *f++;
      *t++ = a; a = *f++;
      *t++ = b; b = *f++;
      *t++ = c; c = *f++;
      *t++ = a; a = *f++;
      *t++ = b; b = *f++;
      *t++ = c; c = *f++;
      *t++ = a; a = *f++;
      *t++ = b;
      *t++ = c;
      *t++ = a;
    }
  gm_assert (t == limit);
}
#undef GM_BCOPY3
