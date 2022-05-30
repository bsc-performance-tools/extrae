/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                   Extrae                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *     ___     This library is free software; you can redistribute it and/or *
 *    /  __         modify it under the terms of the GNU LGPL as published   *
 *   /  /  _____    by the Free Software Foundation; either version 2.1      *
 *  /  /  /     \   of the License, or (at your option) any later version.   *
 * (  (  ( B S C )                                                           *
 *  \  \  \_____/   This library is distributed in hope that it will be      *
 *   \  \__         useful but WITHOUT ANY WARRANTY; without even the        *
 *    \___          implied warranty of MERCHANTABILITY or FITNESS FOR A     *
 *                  PARTICULAR PURPOSE. See the GNU LGPL for more details.   *
 *                                                                           *
 * You should have received a copy of the GNU Lesser General Public License  *
 * along with this library; if not, write to the Free Software Foundation,   *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA          *
 * The GNU LEsser General Public License is contained in the file COPYING.   *
 *                                 ---------                                 *
 *   Barcelona Supercomputing Center - Centro Nacional de Supercomputacion   *
\*****************************************************************************/

#pragma once

# define _GNU_SOURCE
# include <dlfcn.h>

#include "common.h"
#include "pdebug.h"
#include "wrap_checkpoints.h"
#include "change_mode.h"

/*
 * Macros to enable macro specialisation based on an argument, based on:
 * https://stackoverflow.com/questions/11632219/c-preprocessor-macro-specialisation-based-on-an-argument/11640759
 */

#define CAT(a, ...) PRIMITIVE_CAT(a, __VA_ARGS__)
#define PRIMITIVE_CAT(a, ...) a ## __VA_ARGS__

#define IIF(c) PRIMITIVE_CAT(IIF_, c)
#define IIF_0(t, ...) __VA_ARGS__
#define IIF_1(t, ...) t

#define PROBE(x) x, 1

#define CHECK(...) CHECK_N(__VA_ARGS__, 0)
#define CHECK_N(x, n, ...) n

#define TYPE_PROBE(rettype)        TYPE_PROBE_PROXY( TYPE_##rettype )           // concatenate prefix with rettype
#define TYPE_PROBE_PROXY(...)      TYPE_PROBE_PRIMIVIE(__VA_ARGS__)             // expand arguments
#define TYPE_PROBE_PRIMIVIE(x)     TYPE_PROBE_COMBINE_ x                        // merge
#define TYPE_PROBE_COMBINE_(...)   PROBE(~)                                     // if merge successful, expand to probe

// Define new specialised types here
#define TYPE_void ()

#define IS_TYPE(rettype) CHECK(TYPE_PROBE(rettype))

#define FORMAT_PROBE(rettype)      FORMAT_PROBE_PROXY( KNOWN_FORMAT_##rettype ) // concatenate prefix with rettype
#define FORMAT_PROBE_PROXY(...)    FORMAT_PROBE_PRIMIVIE(__VA_ARGS__)           // expand arguments
#define FORMAT_PROBE_PRIMIVIE(x)   FORMAT_PROBE_COMBINE_ x                      // merge
#define FORMAT_PROBE_COMBINE_(...) PROBE(~)                                     // if merge successful, expand to probe

// Define new specialised formats here
#define KNOWN_FORMAT_int ()
#define FORMAT_int "%d"

#define KNOWN_FORMAT_unsigned ()
#define FORMAT_unsigned "%u"

#define IS_KNOWN_FORMAT(rettype) CHECK(FORMAT_PROBE(rettype))


/*
 * Argument manipulation
 */
#define UNPACK_ARG1(arg1, ...) arg1
#define UNPACK_ARG2(arg1, arg2, ...) arg2
#define UNPACK_ARG3(arg1, arg2, arg3, ...) arg3

#define UNPACK_FN UNPACK_ARG1
#define UNPACK_DATA UNPACK_ARG2
#define UNPACK_NUM_THREADS UNPACK_ARG3

// Variable argument list ... can't be empty to prepend/append
#define PREPEND_ARG(first, ...) ARGS(first, __VA_ARGS__)
#define APPEND_ARG(last, ...)   ARGS(__VA_ARGS__, last)

#define REPLACE_ARGS_1_4(new_arg1, new_arg2, new_arg3, new_arg4, old_arg1, old_arg2, old_arg3, old_arg4, ...) \
  ARGS(new_arg1, new_arg2, new_arg3, new_arg4, ## __VA_ARGS__)

#define REPLACE_ARGS_FN_DATA(callback, helper, fn, data, ...) ARGS(callback, helper, ## __VA_ARGS__)


/*
 * Code generation tokens
 */

#define PROTOTYPE(...) (__VA_ARGS__)
#define RETURN(rettype) rettype
#define NO_RETURN RETURN(void)

#define ARGS(...)                   __VA_ARGS__
#define ENTRY_PROBE_ARGS(...)       __VA_ARGS__
#define REAL_SYMBOL_ARGS(...)       __VA_ARGS__
#define EXIT_PROBE_ARGS(...)        __VA_ARGS__
#define DEBUG_ARGS(format, args...) format, ## args
#define VARARGS(...)                __VA_ARGS__

#define REAL_SYMBOL_PTR(symbol) symbol ## _real
#define VARG_SYMBOL_PTR(symbol) symbol ## _varargs

#define DEFAULT_RETURN_VARIABLE_NAME  __xtr_rv__

#define DEFAULT_VARARGS_VARIABLE_NAME __xtr_va__

// Entry/exit probe prefixes 
#define ENTRY_PROBE(symbol) xtr_probe_entry_ ## symbol
#define EXIT_PROBE(symbol)  xtr_probe_exit_ ## symbol

#define ENTRY_PROBE_BURST(symbol) xtr_probe_entry_ ## symbol ## _bursts
#define EXIT_PROBE_BURST(symbol) xtr_probe_exit_ ## symbol ## _bursts

// To pass configurable blocks of code 
#define CODE(...)                    __VA_ARGS__
#define PROLOGUE(...)                __VA_ARGS__
#define EMPTY_PROLOGUE               CODE()
#define EPILOGUE(...)                __VA_ARGS__
#define EMPTY_EPILOGUE               CODE()
#define CODE_BEFORE_ENTRY_PROBE(...) __VA_ARGS__
#define NOOP_BEFORE_ENTRY_PROBE      CODE()
#define CODE_BEFORE_REAL_SYMBOL(...) __VA_ARGS__
#define NOOP_BEFORE_REAL_SYMBOL      CODE()
#define CODE_BEFORE_EXIT_PROBE(...)  __VA_ARGS__
#define NOOP_BEFORE_EXIT_PROBE       CODE()
#define CODE_AFTER_EXIT_PROBE(...)   __VA_ARGS__
#define NOOP_AFTER_EXIT_PROBE        CODE()
#define NOOP                         CODE()

// Decorators for debug messages at wrappers' entry/exit points
#define ENTRY_TAG_BEGIN
#define ENTRY_TAG_END   ">"
#define EXIT_TAG_BEGIN "<"
#define EXIT_TAG_END


/*
 * Code generation macros 
 */

#define GET_3RD_ARG(_0,_1,_2,...) _2

// Pretty prints the value of the wrapper's default return variable for some known rettype's (defined above)
#define GET_FORMAT_FOR_KNOWN_TYPE(rettype) IIF( IS_KNOWN_FORMAT(rettype) ) \
    (\
        " [RV=" FORMAT_ ## rettype "]", "%s" \
    )

#define GET_FORMAT_FOR_KNOWN_TYPE_AT_ENTRY(rettype) "%s"
#define GET_FORMAT_FOR_KNOWN_TYPE_AT_EXIT(rettype) GET_FORMAT_FOR_KNOWN_TYPE(rettype)

// Adds the wrapper's default return variable to the print's argument list for some known retttype's (defined above)
#define GET_RETVAR_FOR_KNOWN_TYPE(rettype) IIF( IS_KNOWN_FORMAT(rettype) ) \
    (\
        DEFAULT_RETURN_VARIABLE_NAME, "" \
    )

#define GET_RETVAR_FOR_KNOWN_TYPE_AT_ENTRY(rettype) ""
#define GET_RETVAR_FOR_KNOWN_TYPE_AT_EXIT(rettype) GET_RETVAR_FOR_KNOWN_TYPE(rettype)

// Changes rettype for an empty token if this is is void; or rettype otherwise
#define RETTYPE_CHOOSER(rettype) IIF( IS_TYPE(rettype) ) \
    (\
        /* nothing */,rettype\
    )

// Changes rettype for an empty token if this is void; or a variable name otherwise
#define RETVAR_CHOOSER(rettype, ...) IIF( IS_TYPE(rettype) ) \
    (\
        /* nothing */,DEFAULT_RETURN_VARIABLE_NAME __VA_ARGS__\
    )

// Changes rettype for an empty token if this is void; or the 'return' keyword otherwise
#define RETURN_CHOOSER(rettype) IIF( IS_TYPE(rettype) ) \
    (\
        /* nothing */,return\
    )

// Writes declaration of a return variable if rettype != void
#define RETVAR_INIT(rettype)   RETTYPE_CHOOSER(rettype) RETVAR_CHOOSER(rettype)

// Writes L-value assigment to a return variable if rettype != void
#define RETVAR_ASSIGN(rettype) RETVAR_CHOOSER(rettype, = )

// Writes return expression of the return variable if rettype != void
#define RETVAR_RETURN(rettype) RETURN_CHOOSER(rettype) RETVAR_CHOOSER(rettype)

// Detect if the last known argument before varargs is passed to conditionally generate va_start & va_end code
#define VARARGS_START_CHOOSER(...) GET_3RD_ARG(_0, ##__VA_ARGS__, VARGS_START1, VARGS_START0)(__VA_ARGS__)
#define VARARGS_END_CHOOSER(...) GET_3RD_ARG(_0, ##__VA_ARGS__, VARGS_END1, VARGS_END0)

#define VARGS_START0(last_argument)
#define VARGS_START1(last_argument)                       \
  va_list DEFAULT_VARARGS_VARIABLE_NAME;                  \
  va_start(DEFAULT_VARARGS_VARIABLE_NAME, last_argument); \

#define VARGS_END0
#define VARGS_END1 va_end(DEFAULT_VARARGS_VARIABLE_NAME);

// Writes va_start declaration if the last known argument before the varargs is passed 
#define VARARGS_START(last_argument) VARARGS_START_CHOOSER(last_argument)

// Writes va_end if the last known argument before the varargs is passed
#define VARARGS_END(last_argument) VARARGS_END_CHOOSER(last_argument)

// Choose the proper way to call to the real symbol depending on the presence of varargs
#define INVOKE_REAL_CHOOSER(...) GET_3RD_ARG(_0, ##__VA_ARGS__, INVOKE_REAL_WITH_VARARGS, INVOKE_REAL)

// Direct call to the real symbol
#define INVOKE_REAL(symbol, rettype, argument_list)                         \
  BYPASS_TO_FUNCTION_PTR(REAL_SYMBOL_PTR(symbol),                           \
                         rettype,                                           \
                         ARGS(argument_list))                               \

/* 
 * Call the real symbol going first through a front-end function (<symbol>_varargs) to parse the variable argument list
 * and select the proper number of arguments to the real symbol. The prototype of this front-end needs to be as follows:
 * - Same return type as the real symbol
 * - The first parameter is the function pointer of the real symbol
 * - All the parameters from the real symbol prototype before the variable argument list 
 * - One last va_list object that represents the variable argument list 
 * Since this snippet is used both from the IF_TRACING_BLOCK's (call the real symbol between the probes), 
 * as well as the BYPASS_BLOCK (tracing disabled), the va_copy is necessary for the first case just in case
 * we also want to pass and unfold the varargs in the probes.
 */
#define INVOKE_REAL_WITH_VARARGS(symbol, rettype, argument_list)              \
  va_list varargs;                                                            \
  va_copy(varargs, DEFAULT_VARARGS_VARIABLE_NAME);                            \
  BYPASS_TO_FUNCTION_PTR(VARG_SYMBOL_PTR(symbol),                             \
                         rettype,                                             \
                         PREPEND_ARG(REAL_SYMBOL_PTR(symbol), argument_list)) \
  va_end(varargs);

// Direct call to the probe
#define INVOKE_PROBE(probe, argument_list)                                    \
 BYPASS_TO_FUNCTION_PTR(probe, void, ARGS(argument_list))

// Direct call to the probe with a private copy of the varargs just in case we also want to pass and unfold the varargs in the probe
#define INVOKE_PROBE_WITH_VARARGS(probe, argument_list)                       \
  va_list varargs;                                                            \
  va_copy(varargs, DEFAULT_VARARGS_VARIABLE_NAME);                            \
  BYPASS_TO_FUNCTION_PTR(probe, void, ARGS(argument_list))                    \
  va_end(varargs);

// Writes a call to the given function_ptr capturing its return value if rettype != void
#define BYPASS_TO_FUNCTION_PTR(function_ptr, rettype, args)                   \
  RETVAR_ASSIGN(rettype) function_ptr (args);                                 \

// Choose the proper template for the IF_TRACING block in the wrapper's body depending on the presence of varargs
#define IF_TRACING_TEMPLATE_CHOOSER(...) GET_3RD_ARG(_0, ##__VA_ARGS__, IF_TRACING_BLOCK_VARARGS, IF_TRACING_BLOCK_NORMAL)

#define IF_TRACING_BLOCK_NORMAL(symbol,                                       \
                                rettype,                                      \
                                condition,                                    \
                                code_before_entry_probe,                      \
                                entry_probe_args,                             \
                                code_before_real_symbol,                      \
                                real_symbol_args,                             \
                                code_before_exit_probe,                       \
                                exit_probe_args,                              \
                                code_after_exit_probe)                        \
if ((EXTRAE_ON()) && (condition))                                             \
{                                                                             \
  ENTERING_INSTRUMENTATION();                                                 \
  code_before_entry_probe                                                     \
  if (ENTRY_PROBE(symbol))                                                    \
  {                                                                           \
    /* Call entry probe */                                                    \
    if (CURRENT_TRACE_MODE(THREADID) == TRACE_MODE_BURST)                    \
    {                                                                         \
      INVOKE_PROBE(ENTRY_PROBE_BURST(symbol), ARGS(entry_probe_args));        \
    }                                                                         \
    else                                                                      \
    {                                                                         \
      INVOKE_PROBE(ENTRY_PROBE(symbol), ARGS(entry_probe_args));              \
    }                                                                         \
  }                                                                           \
  /* Call the runtime and capture the return value */                         \
  code_before_real_symbol                                                     \
  ENTERING_RUNTIME();                                                         \
  INVOKE_REAL(symbol, rettype, ARGS(real_symbol_args));                       \
  EXITING_RUNTIME();                                                          \
  code_before_exit_probe                                                      \
  if (EXIT_PROBE(symbol))                                                     \
  {                                                                           \
    /* Call exit probe */                                                     \
    if (CURRENT_TRACE_MODE(THREADID) == TRACE_MODE_BURST)                    \
    {                                                                         \
      INVOKE_PROBE(EXIT_PROBE_BURST(symbol), ARGS(exit_probe_args));          \
    }                                                                         \
    else                                                                      \
    {                                                                         \
      INVOKE_PROBE(EXIT_PROBE(symbol), ARGS(exit_probe_args));                \
    }                                                                         \
  }                                                                           \
  code_after_exit_probe                                                       \
  EXITING_INSTRUMENTATION();                                                  \
}

#define IF_TRACING_BLOCK_VARARGS(symbol,                                      \
                                 rettype,                                     \
                                 condition,                                   \
                                 code_before_entry_probe,                     \
                                 entry_probe_args,                            \
                                 code_before_real_symbol,                     \
                                 real_symbol_args,                            \
                                 code_before_exit_probe,                      \
                                 exit_probe_args,                             \
                                 code_after_exit_probe)                       \
if ((EXTRAE_ON()) && (condition))                                             \
{                                                                             \
  ENTERING_INSTRUMENTATION();                                                 \
  code_before_entry_probe                                                     \
  if (ENTRY_PROBE(symbol))                                                    \
  {                                                                           \
    /* Call entry probe */                                                    \
    if (CURRENT_TRACE_MODE(THREADID) == TRACE_MODE_BURST)                          \
    {                                                                               \
      INVOKE_PROBE_WITH_VARARGS(ENTRY_PROBE_BURST(symbol), ARGS(entry_probe_args)); \
    }                                                                               \
    else                                                                            \
    {                                                                               \
      INVOKE_PROBE_WITH_VARARGS(ENTRY_PROBE(symbol), ARGS(entry_probe_args));       \
    }                                                                               \
  }                                                                           \
  /* Call the runtime and capture the return value */                         \
  code_before_real_symbol                                                     \
  ENTERING_RUNTIME();                                                         \
  INVOKE_REAL_WITH_VARARGS(symbol, rettype, ARGS(real_symbol_args));          \
  EXITING_RUNTIME();                                                          \
  code_before_exit_probe                                                      \
  if (EXIT_PROBE(symbol))                                                     \
  {                                                                           \
    /* Call exit probe */                                                     \
    if (CURRENT_TRACE_MODE(THREADID) == TRACE_MODE_BURST)                          \
    {                                                                               \
      INVOKE_PROBE_WITH_VARARGS(EXIT_PROBE_BURST(symbol), ARGS(exit_probe_args));   \
    }                                                                               \
    else                                                                            \
    {                                                                               \
      INVOKE_PROBE_WITH_VARARGS(EXIT_PROBE(symbol), ARGS(exit_probe_args));         \
    }                                                                               \
  }                                                                           \
  code_after_exit_probe                                                       \
  EXITING_INSTRUMENTATION();                                                  \
}


#if defined(DEBUG)
/**
 * XTR_WRAP_DBG
 * 
 * Prints by default the address of the real symbol, its parameters, 
 * and the return value for basic data types. Additionally, if the 
 * function xtr_<module>_extra_debug is implemented, it is called 
 * with a buffer to be filled with runtime specific information. 
 */
# define XTR_WRAP_DBG(symbol, module, rettype, where, format, args...) \
{                                                                      \
  char buffer[4096];                                                   \
  xmemset(buffer, 0, sizeof(buffer));                                  \
  if (xtr_ ## module ## _extra_debug != NULL)                          \
  {                                                                    \
    xtr_ ## module ## _extra_debug(buffer, sizeof(buffer));            \
  }                                                                    \
  THREAD_DBG("%s [@=%p]%s%s%s%s" format "%s"                           \
             GET_FORMAT_FOR_KNOWN_TYPE ## _AT_ ## where(rettype) "\n", \
             where ## _TAG_BEGIN #where where ## _TAG_END,             \
             REAL_SYMBOL_PTR(symbol),                                  \
             (strlen(buffer) > 0) ? " [" : "",                         \
             (strlen(buffer) > 0) ? buffer : "",                       \
             (strlen(buffer) > 0) ? "]" : "",                          \
             (strlen(format) > 0 ? " <" : ""),                         \
             ## args,                                                  \
             (strlen(format) > 0 ? ">" : ""),                          \
             GET_RETVAR_FOR_KNOWN_TYPE ## _AT_ ## where(rettype));     \
}
#else
# define XTR_WRAP_DBG(symbol, module, rettype, where, format, args...)
#endif




/**
 * XTR_DEFERRED_SYM_RESOLUTION
 *
 * In case the constructor didn't trigger, or dlsym failed
 * during initialization, retry hooking the current wrapper's
 * real symbol, and quit immediately if not possible.
 *
 * @param real_sym Token of the wrapped symbol
 */
#define XTR_DEFERRED_SYM_RESOLUTION(real_sym)                  \
{                                                              \
  if (REAL_SYMBOL_PTR(real_sym) == NULL)                       \
  {                                                            \
    int success = 0;                                           \
    OMP_HOOK_INIT(real_sym, success);                          \
    if (!success)                                              \
    {                                                          \
      THREAD_ERROR("Wrapper '%s' called but real symbol"       \
                   "pointer is not resolved! EXITING!!!\n",    \
                   __FUNCTION__);                              \
      exit (-1);                                               \
    }                                                          \
  }                                                            \
}


/**
 * XTR_WRAP
 * 
 * Generates wrapper code for OpenMP calls.
 *
 * @param symbol                 Token of the instrumented GOMP symbol (e.g. GOMP_critical_name_start)
 * @param module                 Used for debug information to indentify the group
 * @param prototype              List of input parameters (e.g. 'PROTOTYPE(void **pptr)')
 * @param rettype                Return type of the wrapped symbol (e.g. 'RETTYPE(void)')
 * @param PROLOGUE_BLOCK         Block of code to be executed at the beginning of the wrapper before
 *                               checking if we are going to trace the call
 * @param TRACING_LOGIC_BLOCK    Contains the if block that checks and adds the instrumentation 
 *                               before and after the real call (see IF_TRACING_BLOCK_NORMAL and _VARARGS)
 * @param entry_probe_args       List of arguments passed to the entry probe (all input 
 *                               parameters in 'prototype' are valid) (e.g. 'PROBE_ENTER_ARGS(pptr)' 
 * @param real_symbol_args       List of arguments passed to the real 'fn', all those in 
 *                               'prototype' (e.g. 'REAL_SYMBOL_ARGS(pptr)')
 * @param exit_probe_args        List of arguments passed to the exit probe (all input 
 *                               parameters in 'prototype', as well as 'retvar', are valid) (e.g. 'PROBE_LEAVE_ARGS(pptr)' 
 * @param EPILOGUE_BLOCK         Block of code to be executed at the end, before returning from the wrapper
 * @param debug_format_args      Format string that specifies how subsequent
 *                               arguments are converted for output (e.g. 'DEBUG_FORMAT_ARGS("pptr=%p", pptr)')
 * @param ... [OPTIONAL]         If the 'prototype' contains a varying number of arguments, 
 *                               this specifies the name of the last argument before the variable
 *                               argument list (e.g. 'VARARGS(last)')
 */
#define XTR_WRAP(symbol,                                               \
                 module,                                               \
                 prototype,                                            \
                 rettype,                                              \
                 PROLOGUE_BLOCK,                                       \
                 TRACING_LOGIC_BLOCK,                                  \
                 entry_probe_args,                                     \
                 real_symbol_args,                                     \
                 exit_probe_args,                                      \
                 EPILOGUE_BLOCK,                                       \
                 debug_format_args,                                    \
                 varargs...)                                           \
rettype symbol prototype                                               \
{                                                                      \
  RETVAR_INIT(rettype);                                                \
  VARARGS_START(varargs);                                              \
                                                                       \
  PROLOGUE_BLOCK                                                       \
                                                                       \
  XTR_DEFERRED_SYM_RESOLUTION(symbol);                                 \
  XTR_WRAP_DBG(symbol, module, rettype, ENTRY, debug_format_args);     \
                                                                       \
  /* IF_TRACING BLOCK(s) */                                            \
  TRACING_LOGIC_BLOCK(symbol,                                          \
                      rettype,                                         \
                      ARGS(entry_probe_args),                          \
                      ARGS(real_symbol_args),                          \
                      ARGS(exit_probe_args),                           \
                      IF_TRACING_TEMPLATE_CHOOSER(varargs))            \
  else /* BYPASS BLOCK */                                              \
  {                                                                    \
    INVOKE_REAL_CHOOSER(varargs)(symbol,                               \
                                 rettype,                              \
                                 ARGS(real_symbol_args));              \
  }                                                                    \
                                                                       \
  EPILOGUE_BLOCK                                                       \
                                                                       \
  XTR_WRAP_DBG(symbol, module, rettype, EXIT, debug_format_args);      \
                                                                       \
  VARARGS_END(varargs);                                                \
  RETVAR_RETURN(rettype);                                              \
}
