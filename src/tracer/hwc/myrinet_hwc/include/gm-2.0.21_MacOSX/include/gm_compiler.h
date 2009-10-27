/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2000 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* This file holds compiler abstractions macros. */

#ifndef _gm_compiler_h_
#define _gm_compiler_h_

#include "gm.h"

/****************************************************************
 * Function attributes
 ****************************************************************/

#ifdef __GNUC__
#define __gm_gcc_attribute__(a) __attribute__(a)
#else
#define __gm_gcc_attribute__(a)
#endif

/* Mark a function as potentially unused */

#define GM_FUNCTION_MAY_BE_UNUSED __gm_gcc_attribute__ ((unused))

/****************************************************************
 * printing 64-bit values
 ****************************************************************/

#define GM_U64_TMPL "0x%08x%08x"
#define GM_U64_ARG(x)						\
  ((gm_u32_t) (((gm_u64_t) (x)) >> 31 >> 1) & 0xffffffff),	\
  ((gm_u32_t) (((gm_u64_t) (x)            ) & 0xffffffff))

/****************************************************************
 * duplicate string treatment
 ****************************************************************/

#ifdef __GNUC__
#define GM_COMPILER_COMBINES_IDENTICAL_STRINGS 1
#else
#define GM_COMPILER_COMBINES_IDENTICAL_STRINGS 0
#endif

#endif /* _gm_compiler_h_ */


/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
