/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* This file implements call-tracing hooks:
   
   GM_CALLED(): Report that a function was called.
   GM_TRACE(): Report a message.
   GM_RETURN(val): Report that a function is returning and return;
   GM_RETURN_NOTHING(): Report that a function is returning and return
   	nothing. */

#ifndef _gm_call_trace_h_
#define _gm_call_trace_h_

#include "gm.h"
#include "gm_compiler.h"
#include "gm_cpp.h"
#include "gm_debug.h"
#include "gm_debug_malloc.h"

/****************************************************************
 * Feature switches
 ****************************************************************/

/* Global feature switch: Turn on all tracing. */
/* WARNING - Enabling this in the driver will likely crash SMP systems */

#define GM_ENABLE_CALL_TRACE 0

/* Local feature switch:
   
   Redefine this to 1 IN OTHER FILES to enable tracing despite
   GM_ENABLE_CALL_TRACE.  For example, the following turns on tracing
   for the function FOO() only, but does not effect tracing for other
   functions in the file.

   > ...
   > #undef GM_LOCALLY_ENABLE_CALL_TRACE
   > #define GM_LOCALLY_ENABLE_CALL_TRACE 1
   > void foo (void)
   > {
   >   ...
   > }
   > #undef GM_LOCALLY_ENABLE_CALL_TRACE
   > #define GM_LOCALLY_ENABLE_CALL_TRACE 0
   > ...
*/

#define GM_LOCALLY_ENABLE_CALL_TRACE 0	/* DO NOT CHANGE HERE! */

/****************************************************************
 * Implementation
 ****************************************************************/

GM_ENTRY_POINT const char *__gm_print_called_space (void);
GM_ENTRY_POINT const char *__gm_print_return_space (void);
GM_ENTRY_POINT void gm_call_trace_mutex_enter (void);
GM_ENTRY_POINT void gm_call_trace_mutex_exit (void);

/* Combine the global and local feature switches. */

#define __GM_ENABLE_CALL_TRACE (GM_ENABLE_CALL_TRACE			\
				|| GM_LOCALLY_ENABLE_CALL_TRACE)

/****************
 * Record a description of a function.  This is a submacro for GM_CALLED.
 ****************/

struct gm_function_description
{
  struct gm_function_description *next;
  void *position;
  char *where;
};

extern struct gm_function_description *gm_first_function_description;

#if 0
/* Dead code that may become useful again in the future. */
#define GM_RECORD_FUNCTION_LOCATION() do {				\
  static struct gm_function_description record;				\
  if (!record.where)							\
    {									\
      record.position = GM_CURRENT_PC();				\
      record.where = __GM_FUNCTION__;					\
      record.next = gm_first_function_description;			\
      gm_first_function_description = &record;				\
    }									\
} while (0)
#else  /* 1 */
#define GM_RECORD_FUNCTION_LOCATION()
#endif /* 1 */

/****************
 * Call stack
 *
 * This stack is of limited size and wraps around.  To ensure that
 * false traces are not reported, values are erased as they are
 * popped, allowing the stack to detect when it does not know anything
 * more about the stack trace.
 ****************/

#define GM_CALL_STACK_DEPTH 256

GM_ENTRY_POINT extern char *gm_call_stack[GM_CALL_STACK_DEPTH];
GM_ENTRY_POINT extern void *gm_call_label[GM_CALL_STACK_DEPTH];
GM_ENTRY_POINT extern unsigned long gm_call_depth;
GM_ENTRY_POINT extern int gm_call_stack_overflowed;

/* If possible, this should be an expression that is bogus if
   GM_CALLED() was not invoked in the current function.  It is used so
   that the GM_CALLER macro can verify that GM_CALLED() was invoke so
   that it can provide a reasonable return value. */

#if defined __GNUC__ && !defined __STRICT_ANSI__
#define GM_BOGUS_EXPRESSION_IF_GM_CALLED_NOT_INVOKED()			\
  && invoke_GM_CALLED_first
#else
#define GM_BOGUS_EXPRESSION_IF_GM_CALLED_NOT_INVOKED() 0
#endif

/* return the caller of this function up the specified number of
   levels.  GM_CALLER(0) is the caller of the current function.
   GM_CALLER(1) is the caller of that function, etc.  GM_CALLER(N) is
   valid if GM_CALLER(N-1) is valid, so you must climb the stack to
   insure you don't receive any invalid caller info.  A return value
   of 0 indicates that GM is unable to return the requested caller. */

static gm_inline
void *
__GM_CALLER (unsigned int level)
{
  void *ret;
  
  gm_call_trace_mutex_enter ();
  if (gm_call_depth < level + 2
      || gm_call_depth - level >= GM_NUM_ELEM (gm_call_stack))
    {
      ret = 0;
    }
  else
    {
      ret = gm_call_stack[gm_call_depth - 2 - level];
    }
  gm_call_trace_mutex_exit ();
  return ret;
}

/* This is a HACK to ensure that the caller called GM_CALLED() first. */

#define GM_CALLER(level) 						\
(0 ? GM_BOGUS_EXPRESSION_IF_GM_CALLED_NOT_INVOKED () :			\
 __GM_CALLER (level))


/* This is off by default because it causes kernel crashes on SMP systems,
 * but enable it in the MCP by default when GM_DEBUG set. */
#if GM_DEBUG && GM_MCP
#define GM_ENABLE_CALL_STACK 1
#else
#define GM_ENABLE_CALL_STACK 0
#endif

/* Automatically enable the call stack if GM_DEBUG_MALLOC is set,
   since GM_DEBUG_MALLOC requires it. */

#if GM_DEBUG_MALLOC
#undef GM_ENABLE_CALL_STACK
#define GM_ENABLE_CALL_STACK 1
#endif

/* push the name of the current function onto the call stack */
/* Record the call details. */

#define GM_CALL_STACK_PUSH() do {					\
  if (GM_ENABLE_CALL_STACK)						\
    {									\
      if (gm_call_depth < GM_NUM_ELEM (gm_call_stack))			\
	{								\
	  gm_call_label[gm_call_depth] = (void *)GM_FUNCTION;		\
	  gm_call_stack[gm_call_depth] = __GM_FUNCTION__;		\
	}								\
      gm_call_depth++;							\
    }									\
} while (0)

#define GM_CALL_STACK_CHECK() do {					\
  if (GM_ENABLE_CALL_STACK)						\
    {									\
      if (gm_call_depth-1 >= GM_NUM_ELEM (gm_call_stack))		\
	{								\
	  gm_call_stack_overflowed = 1;					\
	  GM_NOTE_OCCASIONALLY (("GM call stack overflowed.\n"));	\
	}								\
      									\
      /* If possible, Check if an instrumented callee function forgot	\
	 to use GM_RETURN*(). */					\
									\
      if (gm_call_depth							\
	  && gm_call_depth-1 < GM_NUM_ELEM (gm_call_stack)		\
	  && gm_call_label[gm_call_depth-1] != (void *)GM_FUNCTION)	\
	{								\
	  if (!gm_call_stack_overflowed)				\
	    {								\
	      GM_NOTE_ONCE						\
		(("Callee of %s() did not call GM_RETURN*().\n",	\
		  __GM_FUNCTION__));					\
	    }								\
	  								\
	  /* Cleanup the stack by popping any stack entries		\
	     left by the callees. */					\
									\
	  while (gm_call_depth						\
		 && (gm_call_label[gm_call_depth-1]			\
		     != (void *)GM_FUNCTION))				\
	    {								\
	      gm_call_depth--;						\
	    }								\
	}								\
    }									\
} while (0)

/* pop the name off the stack */

#define GM_CALL_STACK_POP() do {					\
  if (GM_ENABLE_CALL_STACK)						\
    {									\
      GM_CALL_STACK_CHECK ();						\
      if (gm_call_depth > 0)						\
	{								\
	  if (gm_call_depth < GM_NUM_ELEM (gm_call_stack))		\
	    {								\
	      gm_call_stack[gm_call_depth] = 0;				\
	    }								\
	  --gm_call_depth;						\
	}								\
    }									\
} while (0)

/****************
 * Oft used strings literals:
 *
 * These are defined as macros so that wherever they are use they
 * are defined *exactly* the same so the compiler can combine
 * duplicate copies of the strings.
 ****************/

#define GM_FUNCTION_CALLED __GM_FUNCTION__ "() called"
#define GM_FUNCTION_RETURNING __GM_FUNCTION__ "() returning"

/****************
 * MCP call logging
 ****************/

#if GM_BUILDING_FIRMWARE && 0
#define GM_MCP_LOG(literal) LOG_DISPATCH (0, literal)
#else
#define GM_MCP_LOG(literal)
#endif

/****************
 * Record the entry into and return from functions.
 ****************/

#define GM_RECORD_FUNCTION_ENTRY() do {					\
  GM_MCP_LOG (GM_FUNCTION_CALLED);					\
  GM_CALL_STACK_PUSH ();						\
  GM_RECORD_FUNCTION_LOCATION ();					\
} while (0)

#define GM_RECORD_FUNCTION_RETURN() do {				\
  GM_MCP_LOG (GM_FUNCTION_RETURNING);					\
  GM_RECORD_FUNCTION_LOCATION ();					\
  GM_CALL_STACK_POP ();							\
} while (0)

/* NOTE: Use of the "if GM_DEBUG" conditions below is predicated on the
 *       assumption that GM_PRINT has no effect unless GM_DEBUG is true.
 *       We employ this conditional in order to keep fussy compilers
 *       like MIPSpro C from complaining about unused variables etc.
 */

/* report that the current function has been called.  The
   GM_CALLED_LABEL is a hack to ensure that no more than one GM_CALLED
   macro is in a function and that a call with a GM_RETURN macro also
   has a GM_CALLED macro. */

#define GM_CALLED_LABEL invoke_GM_CALLED_first
#define GM_REQUIRE_GM_CALLED() do{if(0)goto GM_CALLED_LABEL;}while(0)

#if __GM_ENABLE_CALL_TRACE || GM_ENABLE_CALL_STACK
#define GM_CALLED_EX(gm_flag) GM_CALLED_LABEL: do {			\
  const char *__gm_call_trace_space;					\
									\
  gm_call_trace_mutex_enter ();						\
  GM_RECORD_FUNCTION_ENTRY ();						\
  __gm_call_trace_space = __gm_print_called_space ();			\
  if (__GM_ENABLE_CALL_TRACE || (gm_flag))				\
    {									\
      _GM_PRINT (1, ("%s%s\n",						\
		     __gm_call_trace_space, GM_FUNCTION_CALLED));	\
    }									\
  gm_call_trace_mutex_exit ();						\
} while (0)
#else
#define GM_CALLED_EX(gm_flag) GM_CALLED_LABEL:
#endif

#define GM_CALLED() GM_CALLED_EX (0)

#if __GM_ENABLE_CALL_TRACE || GM_ENABLE_CALL_STACK
#define GM_CALLED_WITH_ARGS_EX(gm_flag, args) GM_CALLED_LABEL: do {	\
  const char *__gm_call_trace_space;					\
									\
  gm_call_trace_mutex_enter ();						\
  GM_RECORD_FUNCTION_ENTRY ();						\
  __gm_call_trace_space = __gm_print_called_space ();			\
  if (__GM_ENABLE_CALL_TRACE || (gm_flag))				\
    {									\
      _GM_PRINT (1, ("%s" __GM_FUNCTION__ "(",				\
		     __gm_call_trace_space));				\
      _GM_PRINT (1, args);						\
      _GM_PRINT (1, (") called\n"));					\
    }									\
  gm_call_trace_mutex_exit ();						\
} while (0)
#else
#define GM_CALLED_WITH_ARGS_EX(gm_flag, args) GM_CALLED_LABEL:
#endif

#define GM_CALLED_WITH_ARGS(args) GM_CALLED_WITH_ARGS_EX (0, args)
     
/* report that the current function is not implemented */

#define GM_NOT_IMP() GM_WARN (("not implemented\n"))

/* record that the current function will return in the next C statement */

#if __GM_ENABLE_CALL_TRACE || GM_ENABLE_CALL_STACK
#define GM_RECORD_RETURN(template, arg) do {				\
  const char *__gm_call_trace_space;					\
									\
  gm_call_trace_mutex_enter ();						\
									\
  GM_REQUIRE_GM_CALLED ();						\
  __gm_call_trace_space = __gm_print_return_space ();			\
  if (__GM_ENABLE_CALL_TRACE)						\
    {									\
      _GM_PRINT (1, ("%s%s" template,					\
		     __gm_call_trace_space, GM_FUNCTION_RETURNING,	\
		     arg));						\
    }									\
  GM_RECORD_FUNCTION_RETURN ();						\
									\
  gm_call_trace_mutex_exit ();						\
} while (0)
#else
#define GM_RECORD_RETURN(template, arg) GM_REQUIRE_GM_CALLED ()
#endif

/* Hack to return a value without triggering an "end-of-loop code not
   reached" warning in the Sun compiler. */

#define _GM_RETURN(x) do {						\
  if (1)								\
    {									\
      return (x);							\
    }									\
} while (0)

/* report that the current function is returning */

#define GM_RETURN(val) do {						\
  GM_RECORD_RETURN ("%s\n", ".");					\
  _GM_RETURN (val);							\
} while (0)

/* report that the current function is returning a DMA ptr, and print
   the return value. */

#define GM_RETURN_DP(val) do {						\
  gm_dp_t gm_return_ret;						\
									\
  gm_return_ret = (val);						\
  GM_RECORD_RETURN (" dma ptr "GM_U64_TMPL"\n",				\
		    GM_U64_ARG (gm_return_ret));			\
  _GM_RETURN (gm_return_ret);						\
} while (0)

/* report that the current function is returning an int, and print
   the return value. */

#define GM_RETURN_INT(val) do {						\
  int gm_return_int;							\
									\
  gm_return_int = (val);						\
  GM_RECORD_RETURN (" %i\n", gm_return_int);				\
  _GM_RETURN (gm_return_int);						\
} while (0)

/* report that the current function is returning a pointer, and print
   the return value. */

#define GM_RETURN_PTR(val) do {						\
  void *gm_return_ptr;							\
									\
  gm_return_ptr = (val);						\
  GM_RECORD_RETURN (" %p\n", gm_return_ptr);				\
  _GM_RETURN (gm_return_ptr);						\
} while (0)

/* report that the current function is returning a status, and print
   the return value. */

#define GM_RETURN_STATUS(val) do {					\
  gm_status_t gm_return_status;						\
									\
  gm_return_status = (val);						\
  if (gm_return_status != GM_SUCCESS)					\
    {									\
      GM_RECORD_RETURN (" with error \"%s\"\n",				\
			gm_strerror (gm_return_status));		\
    }									\
  else									\
    {									\
      GM_RECORD_RETURN (" successfully%s\n", "");			\
    }									\
  _GM_RETURN (gm_return_status);					\
} while (0)

/* report that the current function is returning (without a
   return value). */

#define GM_RETURN_NOTHING() do {					\
  GM_RECORD_RETURN ("%s\n", "");					\
  if (1) /* suppress SUN end-of-loop not reached warning */		\
    {									\
      return;								\
    }									\
} while (0)

/* Print message M if tracing is enabled. */

#define GM_TRACE(m) do {						\
  if (__GM_ENABLE_CALL_TRACE)						\
    {									\
      gm_call_trace_mutex_enter ();					\
      GM_PRINT (0, (": " m "\n"));					\
      GM_MCP_LOG (m);							\
      gm_call_trace_mutex_exit ();					\
    }									\
} while (0)

/****************************************************************
 * prototypes
 ****************************************************************/

void gm_log_call_trace (void);
void _gm_report_memory_leaks (void);

#endif /* _gm_call_trace_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
