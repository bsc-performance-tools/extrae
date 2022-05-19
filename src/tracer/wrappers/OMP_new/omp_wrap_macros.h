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

#include "common.h"

#ifdef HAVE_STDARG_H
# include <stdarg.h>
#endif
#include "nested_id.h"
#include "omp_common.h"
#include "utils.h"
#include "wrap_macros.h"

/**
 * This file contains all the macros to generate code for the wrappers. The code generation is divided in two parts. 
 *
 * XTR_WRAP_*
 *
 * The first part is generated through the XTR_WRAP_* macros (see below). These expand into the skeleton of the 
 * wrapper function, declaration of variables, last resort real symbol resolution, bypass to the runtime when tracing
 * is disabled and return value; and allows adding custom code at these different stages (see XTR_WRAP at wrap_macros.h). 
 * 
 * GOMP_*_TRACING_LOGIC
 *
 * One argument of the XTR_WRAP_* macro is the "logic", which is a nested macro that expands into the IF_TRACING code block 
 * (see GOMP_*_TRACING_LOGIC below). This corresponds to the operations that will be performed by the wrapper only if the 
 * symbol is traced (i.e. enter/exit instrumentation, emit events, hook outlined functions, etc.). This also allows adding 
 * custom code at different stages (e.g. before/after probes, before/after real symbol call). This code block can be 
 * customized with the following arguments:
 *
 * @param symbol                 Token of the instrumented GOMP symbol (e.g. GOMP_critical_name_start)
 * @param rettype                Return type of the wrapped symbol (e.g. 'RETTYPE(void)')
 * @param entry_probe_args       List of arguments passed to the entry probe (all input 
 *                               parameters in 'prototype' are valid) (e.g. 'PROBE_ENTER_ARGS(pptr)')
 * @param real_symbol_args       List of arguments passed to the real 'fn', all those in 
 *                               'prototype' (e.g. 'REAL_SYMBOL_ARGS(pptr)')
 * @param exit_probe_args        List of arguments passed to the exit probe (all input 
 *                               parameters in 'prototype', as well as 'retvar', are valid) (e.g. 'PROBE_LEAVE_ARGS(pptr)')
 * @param IF_TRACING_BLOCK       A sequence of code that checks tracing condition, calls probes and the real symbol, 
 *                               and can be customized at different stages. This argument is either 
 *                               IF_TRACING_BLOCK_NORMAL or _VARARGS (see wrap_macros.h). 
 * 
 * The XTR_WRAP_* macros group all the symbols that share a common logic (e.g. all parallels need to perform the same 
 * instrumentation operations). The idea is to identify all the symbols that behave the same, group them under the 
 * same XTR_WRAP_* macro, and write a common "logic" for them. The customizable arguments at both macro levels allow 
 * to fully customize the wrapper without having to worry about the wrapper control flow. 
 */


/*****************************************************************************\
 * CODE GENERATION MACROS TO IMPLEMENT THE IF_TRACING BLOCK WITHIN A WRAPPER *
\*****************************************************************************/

/**
 * OMP_SET_NUM_THREADS_TRACING_LOGIC
 *
 * Special logic for omp_set_num_threads that executes the tracing block only during sequential code 
 * regardless of the status of the instrumentation, even if tracing is disabled. 
 * See XTR_WRAP_OMP_SET_NUM_THREADS for details.
 */
#define OMP_SET_NUM_THREADS_TRACING_LOGIC( symbol,                \
                                           rettype,               \
                                           entry_probe_args,      \
                                           real_symbol_args,      \
                                           exit_probe_args,       \
                                           IF_TRACING_BLOCK )     \
  IF_TRACING_BLOCK( symbol,                                       \
                    rettype,                                      \
                    /* Only trace if we're in sequential code */  \
                    NOT_IN_PARALLEL,                              \
                    NOOP_BEFORE_ENTRY_PROBE,                      \
                    ARGS(entry_probe_args),                       \
                    NOOP_BEFORE_REAL_SYMBOL,                      \
                    ARGS(real_symbol_args),                       \
                    NOOP_BEFORE_EXIT_PROBE,                       \
                    ARGS(exit_probe_args),                        \
                    NOOP_AFTER_EXIT_PROBE )

/**
 * GOMP_COMMON_TRACING_LOGIC
 *
 * Common logic for all OpenMP wrappers that only have to emit events in the probes.
 */
#define GOMP_COMMON_TRACING_LOGIC( symbol,                                      \
                                   rettype,                                     \
                                   entry_probe_args,                            \
                                   real_symbol_args,                            \
                                   exit_probe_args,                             \
                                   IF_TRACING_BLOCK )                           \
  IF_TRACING_BLOCK( symbol,                                                     \
                    rettype,                                                    \
                    xtr_OMP_check_config(OMP_ENABLED) && (NOT_IN_NESTED),       \
                    NOOP_BEFORE_ENTRY_PROBE,                                    \
                    ARGS(entry_probe_args),                                     \
                    NOOP_BEFORE_REAL_SYMBOL,                                    \
                    ARGS(real_symbol_args),                                     \
                    NOOP_BEFORE_EXIT_PROBE,                                     \
                    ARGS(exit_probe_args),                                      \
                    NOOP_AFTER_EXIT_PROBE )

/**
 * GOMP_OUTLINED_TRACING_LOGIC
 *
 * Common logic for all OpenMP wrappers of the outlined routines/tasks/taskloops.
 * Custom operations for the outlined's are performed at the XTR_WRAP_GOMP_*_OL level.
 * At the IF_TRACING block level, we only emit enter/exit events, and don't check 
 * for tracing enabled. If the outlined has been hooked, this means the tracing
 * was enabled at the instatiation point, thus we want to capture this even if 
 * the tracing is disabled at the execution point.
 */
#define GOMP_OUTLINED_TRACING_LOGIC( symbol,                                    \
                                     rettype,                                   \
                                     entry_probe_args,                          \
                                     real_symbol_args,                          \
                                     exit_probe_args,                           \
                                     IF_TRACING_BLOCK )                         \
  IF_TRACING_BLOCK( symbol,                                                     \
                    rettype,                                                    \
                    (NOT_IN_NESTED),                                            \
                    NOOP_BEFORE_ENTRY_PROBE,                                    \
                    ARGS(entry_probe_args),                                     \
                    NOOP_BEFORE_REAL_SYMBOL,                                    \
                    ARGS(real_symbol_args),                                     \
                    NOOP_BEFORE_EXIT_PROBE,                                     \
                    ARGS(exit_probe_args),                                      \
                    NOOP_AFTER_EXIT_PROBE )

/**
 * GOMP_FORK_TRACING_LOGIC
 * 
 * Common logic for all GOMP_parallel_* wrappers (GCC > 4).
 * Swaps the real outlined function with our instrumentation wrapper.
 *
 * We need to check the current nesting level to decide if we have to intercept 
 * the execution of the outlined function, in any of these cases:
 * - we're in sequential code opening 1st-level parallel,
 * - or in 1st-level parallel and going nested,
 * - or we're already in nested parallelism and this is the master thread in all levels. 
 * 
 * In order to intercept the execution of the outlined function 'fn', we swap 
 * 'fn' to 'callback', where 'callback' is an intermediate wrapper that we provide 
 * that receives the original 'fn' and 'data' packed in a helper data storage, 
 * and serves as a trampoline to the real 'fn' (see XTR_WRAP_GOMP_PARALLEL_OL).
 */
#define GOMP_FORK_TRACING_LOGIC( symbol,                                                                            \
                                 rettype,                                                                           \
                                 entry_probe_args,                                                                  \
                                 real_symbol_args,                                                                  \
                                 exit_probe_args,                                                                   \
                                 IF_TRACING_BLOCK )                                                                 \
  int level = omp_get_level();                                                                                      \
  IF_TRACING_BLOCK( symbol,                                                                                         \
                    rettype,                                                                                        \
                    (xtr_OMP_check_config(OMP_ENABLED) && ( (level < 2) || (TRACE_MASTER_IN_NESTED)) ),             \
                    CODE_BEFORE_ENTRY_PROBE(                                                                        \
                     /* Save the real fn and data, and the helper structure can be local */                         \
                     /* because the fork call has an implicit join                       */                         \
                      struct parallel_helper_t helper =                                                             \
                      PARALLEL_HELPER_INITIALIZER(UNPACK_FN(real_symbol_args), UNPACK_DATA(real_symbol_args));      \
                      if (level == 0)                                                                               \
                      {                                                                                             \
                        /* We're in sequential code opening 1st-level parallelism.           */                     \
                        /* Check if the 'num_threads' clause increases the number of threads */                     \
                        /* inside this parallel block. Only done for 1st level because we    */                     \
                        /* are filtering out all data coming from nested threads.            */                     \
                        OMP_CLAUSE_NUM_THREADS_CHANGE(UNPACK_NUM_THREADS(real_symbol_args));                        \
                      }                                                                                             \
                    ),                                                                                              \
                    ARGS(entry_probe_args),                                                                         \
                    NOOP_BEFORE_REAL_SYMBOL,                                                                        \
                    /* Call our XTR_WRAP_GOMP_PARALLEL_OL and pass real fn/data through helper to it  */            \
                    REAL_SYMBOL_ARGS(REPLACE_ARGS_FN_DATA(symbol ## _OL, &helper, real_symbol_args)),               \
                    NOOP_BEFORE_EXIT_PROBE,                                                                         \
                    ARGS(exit_probe_args),                                                                          \
                    CODE_AFTER_EXIT_PROBE(                                                                          \
                      if (level == 0)                                                                               \
                      {                                                                                             \
                        /* When returning from 1st-level parallelism restore active threads */                      \
                        /* to the previous value of the 'num_threads' clause in this block  */                      \
                        OMP_CLAUSE_NUM_THREADS_RESTORE();                                                           \
                      }                                                                                             \
                    ))

/**
 * GOMP_FORK_START_TRACING_LOGIC
 *
 * Common logic for all GOMP_parallel_*_start wrappers (GCC4 ONLY).
 *
 * We need to check the current nesting level to decide if we have to intercept 
 * the execution of the outlined function, in any of these cases:
 * - we're in sequential code opening 1st-level parallel,
 * - or in 1st-level parallel and going nested,
 * - or we're already in nested parallelism and this is the master thread in all levels. 
 *
 * In GCC4, the fork calls are split in two parts: GOMP_parallel_*_start() and 
 * GOMP_parallel_end(). After the _start() call, the runtime returns immediately 
 * and the _start() wrapper ends, so the helper struct that we use to store 'fn' 
 * and 'data' can not be a local variable, as it would become invalid after the 
 * _start() function ends (see __GOMP_new_helper).
 */
#define GOMP_FORK_START_TRACING_LOGIC( symbol,                                                                      \
                                       rettype,                                                                     \
                                       entry_probe_args,                                                            \
                                       real_symbol_args,                                                            \
                                       exit_probe_args,                                                             \
                                       IF_TRACING_BLOCK )                                                           \
  int level = omp_get_level();                                                                                      \
  IF_TRACING_BLOCK( symbol,                                                                                         \
                    rettype,                                                                                        \
                    (xtr_OMP_check_config(OMP_ENABLED) && ( (level < 2) || (TRACE_MASTER_IN_NESTED)) ),             \
                    CODE_BEFORE_ENTRY_PROBE(                                                                        \
                      if (level == 0)                                                                               \
                      {                                                                                             \
                        /* We're in sequential code opening 1st-level parallelism.           */                     \
                        /* Check if the 'num_threads' clause increases the number of threads */                     \
                        /* inside this parallel block. Only done for 1st level because we    */                     \
                        /* are filtering out all data coming from nested threads.            */                     \
                        OMP_CLAUSE_NUM_THREADS_CHANGE(UNPACK_NUM_THREADS(real_symbol_args));                        \
                      }                                                                                             \
                      /* Save the real fn and data and never free the helper structure because we */                \
                      /* don't know when all threads have exited the parallel                     */                \
                      void *helper = __GOMP_new_helper(UNPACK_FN(real_symbol_args), UNPACK_DATA(real_symbol_args)); \
                    ),                                                                                              \
                    ARGS(entry_probe_args),                                                                         \
                    NOOP_BEFORE_REAL_SYMBOL,                                                                        \
                    /* Call our XTR_WRAP_GOMP_PARALLEL_OL and pass real fn/data through helper to it  */            \
                    REAL_SYMBOL_ARGS(REPLACE_ARGS_FN_DATA(symbol ## _OL, helper, real_symbol_args)),                \
                    NOOP_BEFORE_EXIT_PROBE,                                                                         \
                    ARGS(exit_probe_args),                                                                          \
                    NOOP_AFTER_EXIT_PROBE )

/** 
 * GOMP_FORK_END_TRACING_LOGIC
 *
 * This correspond to the _end() function that closes parallelism (GCC4 ONLY). 
 * Remember that the _end() function only gets executed by the master thread 
 * of that parallel block, hence it's safe to emit events when returning from 
 * 2nd-nesting level (so we trace activity up to 1st-level threads)
 */
#define GOMP_FORK_END_TRACING_LOGIC( symbol,                                                                          \
                                     rettype,                                                                         \
                                     entry_probe_args,                                                                \
                                     real_symbol_args,                                                                \
                                     exit_probe_args,                                                                 \
                                     IF_TRACING_BLOCK )                                                               \
  /* At this point to we are still inside the parallel, so returning_from_level >= 1 */                               \
  int returning_from_level = omp_get_level();                                                                         \
  IF_TRACING_BLOCK( symbol,                                                                                           \
                    rettype,                                                                                          \
                    /* Trace up to 1st-level threads or any nested level for the master */                            \
                    (xtr_OMP_check_config(OMP_ENABLED) && ((returning_from_level <= 2) || (TRACE_MASTER_IN_NESTED))), \
                    NOOP_BEFORE_EXIT_PROBE,                                                                           \
                    ARGS(entry_probe_args),                                                                           \
                    NOOP_BEFORE_REAL_SYMBOL,                                                                          \
                    ARGS(real_symbol_args),                                                                           \
                    NOOP_BEFORE_EXIT_PROBE,                                                                           \
                    ARGS(exit_probe_args),                                                                            \
                    CODE_AFTER_EXIT_PROBE(                                                                            \
                      if (returning_from_level == 1)                                                                  \
                      {                                                                                               \
                        /* When returning from 1st-level parallelism restore active threads */                        \
                        /* to the previous value of the 'num_threads' clause in this block  */                        \
                        OMP_CLAUSE_NUM_THREADS_RESTORE();                                                             \
                      }                                                                                               \
                    ))

/**
 * GOMP_TEAMS_TRACING_LOGIC
 * 
 * This logic implements analogous behaviour to GOMP_FORK_TRACING_LOGIC to instrument the outlined. 
 * The evaluation of the num_threads clause is not considered because this argument does not exist here.
 */
#define GOMP_TEAMS_TRACING_LOGIC( symbol,                                                                      \
                                  rettype,                                                                     \
                                  entry_probe_args,                                                            \
                                  real_symbol_args,                                                            \
                                  exit_probe_args,                                                             \
                                  IF_TRACING_BLOCK )                                                           \
  int level = omp_get_level();                                                                                 \
  IF_TRACING_BLOCK( symbol,                                                                                    \
                    rettype,                                                                                   \
                    (xtr_OMP_check_config(OMP_ENABLED) && ((level < 2) || (TRACE_MASTER_IN_NESTED))),          \
                    CODE_BEFORE_ENTRY_PROBE(                                                                   \
                      /* Save the real fn and data, and the helper structure can be local */                   \
                      /* because the fork call has an implicit join                       */                   \
                      struct parallel_helper_t helper =                                                        \
                      PARALLEL_HELPER_INITIALIZER(UNPACK_FN(real_symbol_args), UNPACK_DATA(real_symbol_args)); \
                    ),                                                                                         \
                    ARGS(entry_probe_args),                                                                    \
                    NOOP_BEFORE_REAL_SYMBOL,                                                                   \
                    /* Call our XTR_WRAP_GOMP_TEAMS_OL and pass real fn/data through helper to it */           \
                    REAL_SYMBOL_ARGS(REPLACE_ARGS_FN_DATA(symbol ## _OL, &helper, real_symbol_args)),          \
                    NOOP_BEFORE_EXIT_PROBE,                                                                    \
                    ARGS(exit_probe_args),                                                                     \
                    NOOP_AFTER_EXIT_PROBE)

/**
 * GOMP_TASK_TRACING_LOGIC
 *
 * Swaps the real outlined task with our instrumentation wrapper.
 *
 * In order to intercept the execution of the outlined task 'fn', we force the execution of the 
 * real GOMP_task changing arguments so that:                        
 * 1) fn -> OL_task, 2) data -> task_helper, 3) cpyfn -> NULL, 4) arg_size -> sizeof(task_helper).
 * When cpyfn != NULL, the runtime builds an argument list that becomes the new data argument of
 * the callback. Since we need to receive 'task_helper' in '', we force cpyfn 
 * to NULL. By doing so, we force the runtime to invoke 'XTR_WRAP_GOMP_TASK_OL' passing 'task_helper' 
 * as data argument. Here we need to perform the same operations that the runtime would do if cpyfn != NULL, 
 * and store the constructed argument list in the helper so that our callback is able to invoke 'fn' 
 * with the proper arguments (see libgomp/task.c on GitHub for reference).
 *
 * Helpers for GOMP_task need to be dynamically allocated because the callback 'XTR_WRAP_GOMP_TASK_OL' 
 * may be executed after GOMP_task has already exited. At the end of 'XTR_WRAP_GOMP_TASK_OL' this helper 
 * can be free'd. 
 *
 * The task_helper has to be passed by reference, with arg_size set to sizeof(task_helper_t *)  
 */  
#define GOMP_TASK_TRACING_LOGIC(symbol,                                                                      \
                                rettype,                                                                     \
                                entry_probe_args,                                                            \
                                real_symbol_args,                                                            \
                                exit_probe_args,                                                             \
                                IF_TRACING_BLOCK)                                                            \
  IF_TRACING_BLOCK( symbol,                                                                                  \
                    rettype,                                                                                 \
                    xtr_OMP_check_config(OMP_ENABLED),                                                       \
                    CODE_BEFORE_ENTRY_PROBE(                                                                 \
                      struct task_helper_t *task_helper = NULL;                                              \
                      task_helper = (struct task_helper_t *) xmalloc(sizeof(struct task_helper_t));          \
                      task_helper->fn = fn;                                                                  \
                      task_helper->buf = xmalloc(sizeof(char) * (arg_size + arg_align - 1));                 \
                      if (cpyfn != NULL)                                                                     \
                      {                                                                                      \
                        char *arg = (char *) ( ((uintptr_t) task_helper->buf + arg_align - 1) &              \
                                              ~((uintptr_t) (arg_align - 1)) );                              \
                       cpyfn (arg, data);                                                                    \
                       task_helper->data = arg;                                                              \
                     }                                                                                       \
                     else                                                                                    \
                     {                                                                                       \
                       memcpy (task_helper->buf, data, arg_size);                                            \
                       task_helper->data = task_helper->buf;                                                 \
                     }                                                                                       \
                     /* Assign a unique id to this task to track when it is scheduled and when executed */   \
                     ATOMIC_COUNTER_INCREMENT(task_helper->id, __GOMP_task_counter, 1);                      \
                   ),                                                                                        \
                   ENTRY_PROBE_ARGS(task_helper),                                                            \
                   NOOP_BEFORE_REAL_SYMBOL,                                                                  \
                    /* Call our XTR_WRAP_GOMP_TASK_OL and pass real fn/data through helper to it  */         \
                   REAL_SYMBOL_ARGS(REPLACE_ARGS_1_4(symbol ## _OL, &task_helper, NULL,                      \
                                                     sizeof(task_helper_t *), real_symbol_args)),            \
                   NOOP_BEFORE_EXIT_PROBE,                                                                   \
                   ARGS(exit_probe_args),                                                                    \
                   NOOP_AFTER_EXIT_PROBE )

/**
 * GOMP_TASKLOOP_TRACING_LOGIC
 *
 * Swaps the real outlined task with our instrumentation wrapper.
 * 
 * In order to intercept the execution of the outlined task 'fn', we swap 
 * 'fn' to 'callback', where 'callback' is an intermediate wrapper that we provide 
 * that receives the original 'fn' and 'data' packed as a trailer behind the original 
 * 'data'. The trailer contains a magic number (0xdeadbeef) to locate where the helper
 * starts by inspecting the original data bit per bit (see XTR_WRAP_GOMP_TASKLOOP_OL).
 */
#define GOMP_TASKLOOP_TRACING_LOGIC( symbol,                                                                    \
                                     rettype,                                                                   \
                                     entry_probe_args,                                                          \
                                     real_symbol_args,                                                          \
                                     exit_probe_args,                                                           \
                                     IF_TRACING_BLOCK )                                                         \
  IF_TRACING_BLOCK( symbol,                                                                                     \
                    rettype,                                                                                    \
                    xtr_OMP_check_config(OMP_ENABLED) && xtr_OMP_check_config(OMP_TASKLOOP_ENABLED),            \
                    CODE_BEFORE_ENTRY_PROBE(                                                                    \
                      struct taskloop_helper_t taskloop_helper;                                                 \
                      long helper_size = sizeof(struct taskloop_helper_t);                                      \
                                                                                                                \
                      /* Magic number to locate the helper in callme_taskloop */                                \
                      taskloop_helper.magicno = (void *)0xdeadbeef;                                             \
                      taskloop_helper.fn = fn;                                                                  \
                                                                                                                \
                      /* Assign a unique id to this taskloop to track when it is scheduled and when executed */ \
                      /* (counter is shared with GOMP_task) */                                                  \
                      ATOMIC_COUNTER_INCREMENT(taskloop_helper.id, __GOMP_task_counter, 1);                     \
                                                                                                                \
                      void *data_trailer = xmalloc(arg_size + helper_size);                                     \
                      /* Append the helper to the end of data */                                                \
                      memcpy (data_trailer, data, arg_size);                                                    \
                      memcpy (data_trailer + arg_size, &taskloop_helper, helper_size);                          \
                    ),                                                                                          \
                    ENTRY_PROBE_ARGS(&taskloop_helper, num_tasks),                                              \
                    NOOP_BEFORE_REAL_SYMBOL,                                                                    \
                    /* Call our XTR_WRAP_GOMP_TASKLOOP_OL and pass real fn/data by appending a trailer to  */   \
                    /* the original data and increasing arg_size correspondingly                           */   \
                    REAL_SYMBOL_ARGS(REPLACE_ARGS_1_4(symbol ## _OL, data_trailer, cpyfn,                       \
                                                      arg_size + helper_size, real_symbol_args)),               \
                    NOOP_BEFORE_EXIT_PROBE,                                                                     \
                    ARGS(exit_probe_args),                                                                      \
                    CODE_AFTER_EXIT_PROBE(                                                                      \
                      /* XXX WARNING! Potential memory leak issue                                   */          \
                      /* Free 'data_trailer' here assuming GOMP_taskloop is a blocking call in the  */          \
                      /* runtime. If this is not the case (untied tasks?), this will likely crash.  */          \
                      /* An alternative would be to free the helper after the last task finishes.   */          \
                      /* For this, we would need a malloc'd counter initialized to 'num_tasks',     */          \
                      /* store in the 'taskloop_helper' both a pointer to this counter as well as   */          \
                      /* the pointer to 'data_trailer', and then at the end of the callback         */          \
                      /* (XTR_WRAP_GOMP_TASKLOOP_OL), decrease the counter atomically and when it   */          \
                      /* reachers zero, free both pointers.                                         */          \
                      xfree(data_trailer);                                                                      \
                    ) )


/*************************************************\
 * CODE GENERATION MACROS TO IMPLEMENT A WRAPPER *
\*************************************************/

/**
 * XTR_WRAP_GOMP*
 *
 * Generates wrapper code that can be customized with the following arguments:
 *
 * @param symbol            Token of the instrumented GOMP symbol (e.g. GOMP_critical_name_start)
 * @param prototype         List of input parameters (e.g. 'PROTOTYPE(void **pptr)')
 * @param rettype           Return type of the wrapped symbol (e.g. 'RETTYPE(void)')
 * @param probe_enter_args  List of arguments passed to the entry probe (all input 
 *                          parameters in 'prototype' are valid) (e.g. 'PROBE_ENTER_ARGS(pptr)' 
 * @param real_symbol_args  List of arguments passed to the real 'fn', all those in 
 *                          'prototype' (e.g. 'REAL_SYMBOL_ARGS(pptr)')
 * @param probe_leave_args  List of arguments passed to the exit probe (all input 
 *                          parameters in 'prototype', as well as 'retvar', are valid) (e.g. 'PROBE_LEAVE_ARGS(pptr)' 
 * @param debug_args        Format string that specifies how subsequent
 *                          arguments are converted for output (e.g. 'DEBUG_FORMAT_ARGS("pptr=%p", pptr)')
 * @param ... [OPTIONAL]    If the 'prototype' contains a varying number of arguments, 
 *                          this specifies the name of the last argument before the variable
 *                          argument list (e.g. 'VARARGS(last)')
 */
#define XTR_WRAP_GOMP(symbol,           \
                      prototype,        \
                      rettype,          \
                      probe_enter_args, \
                      real_symbol_args, \
                      probe_leave_args, \
                      debug_args,       \
                      varargs...)       \
XTR_WRAP(symbol, GOMP, prototype, rettype, EMPTY_PROLOGUE, GOMP_COMMON_TRACING_LOGIC, ARGS(probe_enter_args), ARGS(real_symbol_args), ARGS(probe_leave_args), EMPTY_EPILOGUE, ARGS(debug_args), ## varargs)

#define XTR_WRAP_GOMP_TASK(symbol,           \
                           prototype,        \
                           rettype,          \
                           probe_enter_args, \
                           real_symbol_args, \
                           probe_leave_args, \
                           debug_args,       \
                           varargs...)       \
XTR_WRAP(symbol, GOMP, prototype, rettype, EMPTY_PROLOGUE, GOMP_TASK_TRACING_LOGIC, ARGS(probe_enter_args), ARGS(real_symbol_args), ARGS(probe_leave_args), EMPTY_EPILOGUE, ARGS(debug_args), ## varargs)

#define XTR_WRAP_GOMP_TASKLOOP(symbol,           \
                               prototype,        \
                               rettype,          \
                               probe_enter_args, \
                               real_symbol_args, \
                               probe_leave_args, \
                               debug_args)       \
XTR_WRAP(symbol, GOMP, prototype, rettype, EMPTY_PROLOGUE, GOMP_TASKLOOP_TRACING_LOGIC, ARGS(probe_enter_args), ARGS(real_symbol_args), ARGS(probe_leave_args), EMPTY_EPILOGUE, ARGS(debug_args) )

#define XTR_WRAP_GOMP_FORK(symbol,      \
                      prototype,        \
                      rettype,          \
                      probe_enter_args, \
                      real_symbol_args, \
                      probe_leave_args, \
                      debug_args)       \
XTR_WRAP(symbol, GOMP, prototype, rettype, EMPTY_PROLOGUE, GOMP_FORK_TRACING_LOGIC, ARGS(probe_enter_args), ARGS(real_symbol_args), ARGS(probe_leave_args), EMPTY_EPILOGUE, ARGS(debug_args))

#define XTR_WRAP_GOMP_TEAMS(symbol,      \
                      prototype,        \
                      rettype,          \
                      probe_enter_args, \
                      real_symbol_args, \
                      probe_leave_args, \
                      debug_args)       \
XTR_WRAP(symbol, GOMP, prototype, rettype, EMPTY_PROLOGUE, GOMP_TEAMS_TRACING_LOGIC, ARGS(probe_enter_args), ARGS(real_symbol_args), ARGS(probe_leave_args), EMPTY_EPILOGUE, ARGS(debug_args))

/**
 * XTR_WRAP_GOMP_FORK_START
 * XTR_WRAP_GOMP_FORK_END
 *
 * Generates wrapper code for OpenMP forking calls (GCC4 ONLY). 
 */
#define XTR_WRAP_GOMP_FORK_START(symbol,   \
                      prototype,        \
                      rettype,          \
                      probe_enter_args, \
                      real_symbol_args, \
                      probe_leave_args, \
                      debug_args)       \
XTR_WRAP(symbol, GOMP, prototype, rettype, EMPTY_PROLOGUE, GOMP_FORK_START_TRACING_LOGIC, ARGS(probe_enter_args), ARGS(real_symbol_args), ARGS(probe_leave_args), EMPTY_EPILOGUE, ARGS(debug_args))

#define XTR_WRAP_GOMP_FORK_END(symbol,  \
                      prototype,        \
                      rettype,          \
                      probe_enter_args, \
                      real_symbol_args, \
                      probe_leave_args, \
                      debug_args)       \
XTR_WRAP(symbol, GOMP, prototype, rettype, EMPTY_PROLOGUE, GOMP_FORK_END_TRACING_LOGIC, ARGS(probe_enter_args), ARGS(real_symbol_args), ARGS(probe_leave_args), EMPTY_EPILOGUE, ARGS(debug_args))

/**
 * XTR_WRAP_GOMP_PARALLEL_OL
 *
 * Generates wrapper code for outlined parallels. 
 * This corresponds to the callback that we have inserted to replace the
 * original outlined function, and will be invoked by the runtime when
 * it tries to execute the outlined.
 * 
 * At this point we need to recover the pointer to the original outlined
 * and invoke it with its original data arguments. Both arguments can 
 * be retrieved from the helper structure that we also injected as new
 * data argument (*_helper_ptr). Thus, the real outlined can be invoked 
 * through '*_helper_ptr->fn' passing '*_helper_ptr->data' as argument.
 *  
 * Like any other wrapper generated through XTR_WRAP, it expands into an 
 * IF_TRACING block that injects probes around the execution of the outlined.
 *
 * The jump to the real outlined is implicit through XTR_WRAP, which  
 * assumes a function pointer named REAL_SYMBOL_PTR(symbol) (i.e. 
 * GOMP_parallel_OL_real, GOMP_task_OL_real, GOMP_taskloop(_ull)_OL_real)
 * is declared and invoked. Since there may be multiple concurrent calls 
 * to this callback, each jumping to different outlined functions, the 
 * function pointer needs to be declared locally in the PROLOGUE section.
 */
#define XTR_WRAP_GOMP_PARALLEL_OL(symbol)                                                    \
  XTR_WRAP( symbol,                                                                          \
            GOMP,                                                                            \
            PROTOTYPE(void *par_helper_ptr),                                                 \
            NO_RETURN,                                                                       \
            PROLOGUE(                                                                        \
              void (*REAL_SYMBOL_PTR(symbol)) (void *) = NULL;                               \
                                                                                             \
              struct parallel_helper_t *par_helper = par_helper_ptr;                         \
                                                                                             \
              if ((par_helper == NULL) || (par_helper->fn == NULL))                          \
              {                                                                              \
                THREAD_ERROR("Invalid helper");                                              \
                exit(-1);                                                                    \
              }                                                                              \
              /* Set GOMP_parallel_OL_real function pointer to the real outlined */          \
              REAL_SYMBOL_PTR(symbol) = par_helper->fn;                                      \
            ),                                                                               \
          GOMP_OUTLINED_TRACING_LOGIC,                                                       \
          ENTRY_PROBE_ARGS(par_helper),                                                      \
          REAL_SYMBOL_ARGS(par_helper->data), /* Pass original data */                       \
          EXIT_PROBE_ARGS(),                                                                 \
          EMPTY_EPILOGUE,                                                                    \
          DEBUG_ARGS("data_helper_ptr=%p fn=%p data=%p",                                     \
                      par_helper, par_helper->fn, par_helper->data) )


/**
 * XTR_WRAP_GOMP_TASK_OL
 *
 * Generates wrapper code for outlined tasks. 
 * Analogous to XTR_WRAP_GOMP_PARALLEL_OL, with the difference that we need to free the 
 * helper at the end of the wrapper (see EPILOGUE).
*/
#define XTR_WRAP_GOMP_TASK_OL(symbol)                                                        \
  XTR_WRAP( symbol,                                                                          \
            GOMP,                                                                            \
            PROTOTYPE(void *task_helper_ptr),                                                \
            NO_RETURN,                                                                       \
            PROLOGUE(                                                                        \
              void (*REAL_SYMBOL_PTR(symbol)) (void *) = NULL;                               \
              struct task_helper_t *task_helper = *(struct task_helper_t **)task_helper_ptr; \
              if ((task_helper == NULL) || (task_helper->fn == NULL))                        \
              {                                                                              \
                THREAD_ERROR("Invalid helper");                                              \
                exit(-1);                                                                    \
              }                                                                              \
              /* Set GOMP_task_OL_real function pointer to the real outlined */              \
              REAL_SYMBOL_PTR(symbol) = task_helper->fn;                                     \
            ),                                                                               \
            GOMP_OUTLINED_TRACING_LOGIC,                                                     \
            ENTRY_PROBE_ARGS(task_helper),                                                   \
            REAL_SYMBOL_ARGS(task_helper->data), /* Pass original data */                    \
            EXIT_PROBE_ARGS(),                                                               \
            EPILOGUE(                                                                        \
              if (task_helper != NULL)                                                       \
              {                                                                              \
                if (task_helper->buf != NULL)                                                \
                {                                                                            \
                  free(task_helper->buf);                                                    \
                }                                                                            \
                free(task_helper);                                                           \
              }                                                                              \
            ),                                                                               \
            DEBUG_ARGS("task_helper_ptr=%p", task_helper_ptr) )

/**
 * XTR_WRAP_GOMP_TASKLOOP_OL
 * 
 * Generates wrapper code for outlined tasks. 
 * In taskloops, the helper does not arrive here as a new data argument, but as a trailer
 * in the original. We need to locate where the trailer starts by finding a magic number.
 */
#define XTR_WRAP_GOMP_TASKLOOP_OL(symbol)                                                    \
  XTR_WRAP( symbol,                                                                          \
            GOMP,                                                                            \
            PROTOTYPE(void *data_trailer),                                                   \
            NO_RETURN,                                                                       \
            PROLOGUE(                                                                        \
              void (*REAL_SYMBOL_PTR(symbol)) (void *) = NULL;                               \
              struct taskloop_helper_t *taskloop_helper = NULL;                              \
              /* Search for the magic number 0xdeadbeef to locate the helper */              \
              int i = sizeof(void *), arg_size = 0;                                          \
              while ( *(void **)(data_trailer + i) != (void *)0xdeadbeef )                   \
              {                                                                              \
                i ++;                                                                        \
              }                                                                              \
              arg_size = i;                                                                  \
              taskloop_helper = data_trailer + arg_size;                                     \
                                                                                             \
              /* Set GOMP_taskloop(_ull)_OL_real function pointer to the real outlined */    \
              REAL_SYMBOL_PTR(symbol) = taskloop_helper->fn;                                 \
            ),                                                                               \
            GOMP_OUTLINED_TRACING_LOGIC,                                                     \
            ENTRY_PROBE_ARGS(taskloop_helper),                                               \
            REAL_SYMBOL_ARGS(data_trailer),                                                  \
            EXIT_PROBE_ARGS(),                                                               \
            EMPTY_EPILOGUE,                                                                  \
            DEBUG_ARGS("taskloop_helper=%p fn=%p data=%p",                                   \
                       taskloop_helper, taskloop_helper->fn, data_trailer) )

/**
 * XTR_WRAP_GOMP_TEAMS_OL
 * 
 * Outlineds in teams work exactly the same as those for parallels. 
 */
#define XTR_WRAP_GOMP_TEAMS_OL(symbol) XTR_WRAP_GOMP_PARALLEL_OL(symbol)


/**
 * XTR_WRAP_OMP_SET_NUM_THREADS
 *
 * Generates the wrapper code for omp_set_num_threads to allocate more buffers, if needed.
 * 
 * The call to omp_set_num_threads will affect the next parallel region (not the current one), 
 * but by the time the next GOMP_parallel (or similar) arrives, if it does not contain an explicit 
 * 'num_threads' clause, we can not know how many threads will be opened by the runtime unless we 
 * capture this information beforehand from this wrapper. 
 * 
 * As we only represent rows for the 1st-level parallelism, we don't want to capture this call 
 * within a parallel region, because tracing events triggered from nested (non-master) threads 
 * are discarded, thus we don't care about how many threads are used in nested and don't need buffers for them. 
 * 
 * And this needs to be done regardless the status of the instrumentation, even if it is currently disabled, 
 * to have the buffers ready for subsequent parallels when the tracing is restarted.
 * 
 * For details on the tracing conditions for this wrapper, and the buffer reallocation operation, 
 * see OMP_SET_NUM_THREADS_TRACING_LOGIC and xtr_probe_exit_omp_set_num_threads, respectively. 
 */
#define XTR_WRAP_OMP_SET_NUM_THREADS(symbol)                                              \
XTR_WRAP(symbol, GOMP, PROTOTYPE(int num_threads), NO_RETURN,                             \
         EMPTY_PROLOGUE,                                                                  \
         OMP_SET_NUM_THREADS_TRACING_LOGIC,                                               \
         ENTRY_PROBE_ARGS(), REAL_SYMBOL_ARGS(num_threads), EXIT_PROBE_ARGS(num_threads), \
         EMPTY_EPILOGUE,                                                                  \
         DEBUG_ARGS("num_threads=%d", num_threads));
