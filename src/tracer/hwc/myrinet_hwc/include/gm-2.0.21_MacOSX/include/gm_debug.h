/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

#ifndef _gm_debug_h_
#define _gm_debug_h_

#include "gm.h"
#include "gm_config.h"		/* grok the architecture */
#include "gm_cpp.h"
#include "gm_enable_print.h"

#if GM_BUILDING_FIRMWARE
/*#include <stdio.h>*/
#endif

#if GM_OS_VXWORKS
#include <stdio.h>
#endif

/****************************************************************
 * globals
 ****************************************************************/

extern int gm_debug_printing;

/***********************************************************************
 * macros and other defines to support debugging
 ***********************************************************************/

/* GM_STATIC
 *
 * Under Irix, the kernel debugging tools can't "see" even the
 * names of functions that are declared 'static'.  It's useful to have
 * those functions be static if we are not debugging, but when debugging,
 * it's more important to have them visible in a crash dump.
 *
 * However, some strict compilers will generate warnings for any
 * non-static function that does not have a prototype.  Therefore, the
 * default should be to have GM_STATIC defined to "static".  This
 * useful compiler feature forces programmers to either declare their
 * functions static (which enables the compiler to optimize away or
 * warn of unused functions in a source file) or include the .h file
 * that declares the functions prototype (which ensures that the
 * function and its prototype match, preventing insidious stack bugs).
 * Therefore, this compiler feature should be used whenever possible,
 * and the default definition for GM_STATIC should be
 * "static". --Glenn */

#if GM_DEBUG && GM_OS_IRIX && GM_KERNEL
# define GM_STATIC
#else   /* ! GM_DEBUG */
# define GM_STATIC static
#endif  /* ! GM_DEBUG */

/***********************************************************************
 * printing
 ***********************************************************************/

/* gm_printf_p: print message if debugging, unless running on LANai.
   Also prints file, line, and function if possible. */

#if GM_MCP && GM_PRINT_LEVEL >= 10 && defined __GNUC__
#define gm_printf_p(args...) gm_printf (args)
#elif GM_MCP && GM_PRINT_LEVEL < 10 && defined __GNUC__
#define gm_printf_p(args...)
#elif !GM_MCP && GM_PRINT_LEVEL >=10 && defined __GNUC__
#define gm_printf_p(args) do						\
{									\
  if (gm_debug_printing)						\
    {									\
      gm_printf (__GM_WHERE__":");					\
      gm_printf (args);							\
    }									\
} while (0)
#elif !GM_MCP && !defined __GNUC__
#define gm_printf_p(args) printf (args)
#elif !GM_MCP && defined __GNUC__ && defined __STRICT_ANSI__
#define gm_printf_p printf
#elif !GM_MCP && defined __GNUC__ && !defined __STRICT_ANSI__
#define gm_printf_p(args...) printf (args)
#else
#error Do not know how to gm_printf_p()
#endif

/***********************************************************************
 * flashing the LED
 ***********************************************************************/

#if GM_MCP
void gm_morse_async (char *s);
void gm_morse_sync (char *s);
#endif /* GM_MCP */

/***********************************************************************
 * assertions
 ***********************************************************************/

#define _GM_PRINT_PREFIX __GM_WHERE__ "()"

#if GM_KERNEL
#  define GM_PRINT_PREFIX _GM_PRINT_PREFIX ":kernel"
#elif GM_BUILDING_FIRMWARE
#  define GM_PRINT_PREFIX _GM_PRINT_PREFIX ":firmware"
#else
#  define GM_PRINT_PREFIX _GM_PRINT_PREFIX ":userland"
#endif

/****************
 * GM_ARCH_PRINT, INFO, PANIC, NOTE, WARN definitions
 ****************/

#if !GM_KERNEL
#define GM_ARCH_INFO(args) gm_eprintf args
#define GM_ARCH_NOTE(args) gm_eprintf args
#define GM_ARCH_PANIC(args) gm_eprintf args
#define GM_ARCH_PRINT(args) gm_eprintf args
#define GM_ARCH_WARN(args) gm_eprintf args
#else  /* GM_KERNEL */
#include "gm_arch.h"
#endif /* GM_KERNEL */

/****************
 * Print ontinuation macros
 ****************/

/* Use these to continue a GM_INFO, GM_NOTE, etc. message without extra
   labelling.  For example,
   GM_WARN  ("****************\n");
   _GM_WARN ("* example\n");
   _GM_WARN ("****************\n"); */

#define _GM_INFO(args) GM_ARCH_INFO (args)
#define _GM_NOTE(args) GM_ARCH_NOTE (args)
#define _GM_PANIC(args) GM_ARCH_PANIC (args)
#define _GM_WARN(args) GM_ARCH_WARN (args)

#if GM_ENABLE_PRINT || GM_DEBUG
#define _GM_PRINT(flag, args) do{if(flag){GM_ARCH_PRINT(args);}}while(0)
#else
#define _GM_PRINT(flag, args) __gm_check_syntax (flag)
#endif

/****************
 * 
 ****************/

#ifndef GM_PRINT_LEVEL
#define GM_PRINT_LEVEL 0
#endif

/* NOTE: The following are macros instead of inline functions so that
   these macros can report the __GM_FILE__ and __LINE__ of the caller via
   GM_PRINT_PREFIX. */

#if GM_ENABLE_PRINT || GM_DEBUG
#define GM_PRINT(flag, args)						\
do {									\
  int gm_print_flag;							\
									\
  gm_print_flag = (flag) ? 1 : 0;					\
  _GM_PRINT (gm_print_flag, (GM_PRINT_PREFIX ":" #flag ":\n"));		\
  _GM_PRINT (gm_print_flag, args);					\
} while (0)
#else
#define GM_PRINT(flag, args) __gm_check_syntax (flag)
#endif

#define GM_INFO(args) GM_ARCH_INFO (args)
  
#define GM_PANIC(args)							\
do {									\
  _GM_PANIC (("PANIC: "GM_PRINT_PREFIX "\n"));				\
  _GM_PANIC (args);							\
  gm_abort ();								\
} while (0)

#define GM_NOTE(args)							\
do {									\
  _GM_NOTE (( "NOTICE: "GM_PRINT_PREFIX "\n"));				\
  _GM_NOTE (args);							\
} while (0)

#define GM_WARN(args)							\
do {									\
  _GM_WARN (("WARNING: "GM_PRINT_PREFIX "\n"));				\
  _GM_WARN (args);							\
} while (0)

/* Same as GM_NOTE(), but only print one time. */

#define GM_NOTE_ONCE(args)						\
do {									\
  static unsigned long GM_NOTE_ONCE_done;				\
  if (!GM_NOTE_ONCE_done)						\
    {									\
      GM_NOTE (args);							\
      GM_NOTE_ONCE_done = 1;						\
    }									\
} while (0)

/* Same as GM_NOTE(), but only print the message when the number of
   occurences of the event being reported has increased by an order of
   magnitude since the last print. */

#define GM_NOTE_OCCASIONALLY(args)					\
do {									\
  static unsigned long GM_NOTE_OCCASIONALLY_count;			\
  if (GM_POWER_OF_TWO (GM_NOTE_OCCASIONALLY_count))			\
    {									\
      GM_NOTE (args);							\
      if (GM_NOTE_OCCASIONALLY_count/2 > 1)				\
	{								\
	  _GM_NOTE (("last message repeated %lu times\n",		\
		     GM_NOTE_OCCASIONALLY_count/2));			\
	}								\
      GM_NOTE_OCCASIONALLY_count++;					\
    }									\
} while (0)

#endif /* _gm_debug_h_ */

/*
  This file uses GM standard indentation.

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
