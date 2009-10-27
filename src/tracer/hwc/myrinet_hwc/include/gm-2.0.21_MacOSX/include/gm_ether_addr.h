/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2000 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_ether_addr_h_
#define _gm_ether_addr_h_

/* Handy macros to printf ethernet MAC addresses. */

#define GM_MAC_TMPL "%02x:%02x:%02x:%02x:%02x:%02x"
#define _GM_MAC_ARG(m) m[0], m[1], m[2], m[3], m[4], m[5]
#define GM_MAC_ARG(m) _GM_MAC_ARG((unsigned int) (m))

/* ethernet address comparison designed for efficient pipelining. */

static inline int
ether_addr_cmp (gm_s8_t * a, gm_s8_t * b)
{
  int ret, c, d, e, f;

  c = a[0];
  d = b[0];
  e = a[1];
  f = b[1];
  if ((ret = c - d) == 0)
    {
      c = a[2];
      d = b[2];
      if ((ret = e - f) == 0)
	{
	  e = a[3];
	  f = b[3];
	  if ((ret = c - d) == 0)
	    {
	      c = a[4];
	      d = b[4];
	      if ((ret = e - f) == 0)
		{
		  e = a[5];
		  f = b[5];
		  if ((ret = c - d) == 0)
		    {
		      ret = e - f;
		    }
		}
	    }
	}
    }
  return ret;
}

static inline void
ether_addr_copy (gm_u8_t * a, gm_u8_t * b)
{
  unsigned int c, d, e;

  c = a[0];
  d = a[1];
  e = a[2];
  b[0] = c;
  b[1] = d;
  b[2] = e;
  c = a[3];
  d = a[4];
  e = a[5];
  b[3] = c;
  b[4] = d;
  b[5] = e;
}

#endif /* _gm_ether_addr_h_ */
