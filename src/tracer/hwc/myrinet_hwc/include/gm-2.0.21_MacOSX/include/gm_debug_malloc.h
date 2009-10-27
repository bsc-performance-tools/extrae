/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1998 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_debug_malloc_h_
#define _gm_debug_malloc_h_

#include "gm_config.h"

/* Malloc debugging now performs only constant time checks, so this can
   be used any time GM_DEBUG is turned on.  However the memory overhead
   is fairly high. */

#define GM_DEBUG_MALLOC 0

/****************************************************************
 * If building GM C code, prevent calls to malloc(), free(),
 * and calloc() so that gm_malloc() et al. can be used to
 * debug memory leaks.  This is done by defining malloc these
 * function names to a bogus value that the linker will not be
 * able to resolve.
 ****************************************************************/

/****************
 * Macro to informatively rename a function that should not be used.
 * The eventual linker error should specify the location of the
 * prevented reference.
 ****************/

#define __GM_DO_NOT_USE(func, line)					\
  use_gm_ ## func ## _not_ ## func ## _on_line_ ## line
#define _GM_DO_NOT_USE(func, line)					\
  __GM_DO_NOT_USE (func, line)
#define GM_DO_NOT_USE(func) _GM_DO_NOT_USE (func, __LINE__)

/****************
 * Determine whether to rename malloc() et al.
 ****************/

#if GM_BUILDING_GM && !defined (__cplusplus) && GM_DEBUG_MALLOC
#if GM_OS_AIX
/****************
 * Redefining malloc() as a macro doesn't work in AIX, because AIX has
 * already defined malloc() as a macro, and the order of header file
 * inclusion can't reasonably be coerced to make the AIX header file come
 * first (in which case, we could undef the macro here before defining it).
 ****************/
#define GM_PREVENT_NATIVE_MALLOC_USE 0
#else
#define GM_PREVENT_NATIVE_MALLOC_USE 1
#endif
#else
#define GM_PREVENT_NATIVE_MALLOC_USE 0
#endif

/****************
 * Redefine malloc() et al. to ensure they are never called.
 *
 * IF THESE CAUSE ERRORS DURING YOUR COMPILE, DO NOT CHANGE THIS.
 * Instead change your program to call gm_malloc() et al.  Or, if you
 * must use the functions directly (for example to implement
 * __gm_arch_kernel_malloc()), then see how this is done in
 * drivers/vxworks/gm/gm_arch.c or libgm/gm_malloc.c.
 ****************/

#if GM_PREVENT_NATIVE_MALLOC_USE
#define malloc GM_DO_NOT_USE (malloc)
#define calloc GM_DO_NOT_USE (calloc)
#define free GM_DO_NOT_USE (free)
#endif /* GM_PREVENT_MALLOC_USE */

#endif /* _gm_debug_malloc_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
