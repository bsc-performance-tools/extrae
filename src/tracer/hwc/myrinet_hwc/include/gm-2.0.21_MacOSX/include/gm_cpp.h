/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2000 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* this file includes commonly used preprocessor hacks. */

#ifndef _gm_cpp_h_
#define _gm_cpp_h_

/****************************************************************
 * Preprocessor utility macros
 ****************************************************************/

#define GM_STRINGIFY(param) #param
#define GM_SUB_N_STRINGIFY(param) GM_STRINGIFY (param)

#define GM_CAT(a,b) a ## b
#define GM_CAT3(a,b,c) a ## b ## c
#define GM_CAT4(a,b,c,d) a ## b ## c ## d

#define GM_SUB_N_CAT(a,b) GM_CAT(a,b)
#define GM_SUB_N_CAT3(a,b,c) GM_CAT3(a,b,c)
#define GM_SUB_N_CAT4(a,b,c,d) GM_CAT4(a,b,c,d)

/****************************************************************
 * Debugging strings
 ****************************************************************/

/* __GM_LINE__ : The current line being compile AS A STRING.  Use
   the ANSI standard "__LINE__" macro for the line as an int. */

#define __GM_LINE__ GM_SUB_N_STRINGIFY (__LINE__)

/* __GM_UNKNOWN_FUNCTION__ : A description of a function when
   the name of the function is unknown.  This is used by the
   gm_mark_functions shell script. */

#define __GM_UNKNOWN_FUNCTION__ "(unknown function)"


/* __GM_WHERE__ : the current compilation position, with as much
   detail as is available, in a format compatible with emacs
   compilation mode for automatic editor jumping to the line. */

#define __GM_WHERE__ __GM_FILE__ ":" __GM_LINE__ ":" __GM_FUNCTION__

/****************************************************************
 * Compile-time checks
 ****************************************************************/

#ifdef __GNUC__
#define GM_TOP_LEVEL_ASSERT(cond)					\
extern gm_u8_t __GM_TOP_LEVEL_ASSERT[(cond) ? 1 : -1]
#else
#define GM_TOP_LEVEL_ASSERT(cond) struct gm_hack_to_ignore_top_level_semicolon
#endif

/****************************************************************
 * #warning support
 ****************************************************************/

/* Determine if we should use #error's instead of #warnings.  For
   example, #warning generates an error on the SGI Mips Pro compiler,
   but #error just generates a warning.  (Crazy, Huh?) */
#if defined sgi && defined _COMPILER_VERSION	/* SGI Mips Pro compiler */
#define GM_CPP_ERRORS_BEHAVE_LIKE_WARNINGS 1
#else
#define GM_CPP_ERRORS_BEHAVE_LIKE_WARNINGS 0
#endif

#if __GNUC__
#ifndef GM_CPP_WARNINGS_BEHAVE	/* allow Makefile override to disable */
#define GM_CPP_WARNINGS_BEHAVE 1
#endif /* GM_CPP_WARNINGS_BEHAVE */
#else
#define GM_CPP_WARNINGS_BEHAVE 0
#endif

#endif /* _gm_cpp_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
