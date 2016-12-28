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

#include "common.h"

#if HAVE_UNISTD_H
# include <unistd.h>
#endif
#if HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#if HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#if HAVE_PTHREAD_H
# include <pthread.h>
#endif

#include "threadid.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "syscall_probe.h"

/***********************************************************************************************\
 * This file contains the probes to record the begin/end events for instrumented system calls. *
 * As usual, the probes are separate for begin and end events to be able to be injected before *
 * and after the call site or at the entry and exit points of the instrumented routine with    *
 * Dyninst. Despite the support, the system calls are not currently intercepted with Dyninst.  *
\***********************************************************************************************/

/* Global variable to control whether the tracing for I/O calls is enabled */
static int trace_syscall_enabled = FALSE;

/** 
 * Extrae_set_trace_syscall
 * 
 * \param enable Set the tracing for I/O calls enabled or disabled.
 */
void Extrae_set_trace_syscall (int enable)
{
  trace_syscall_enabled = enable; 
}

/** 
 * Extrae_get_trace_syscall
 *
 * \return true if the tracing for I/O calls is enabled; false otherwise.
 */
int Extrae_get_trace_syscall (void)
{ 
  return trace_syscall_enabled; 
}

/**
 * Probe_SYSCALL_sched_yield_Entry
 *
 * Probe injected at the beginning of the syscall 'sched_yield' 
 */
void Probe_SYSCALL_sched_yield_Entry ()
{
  if (mpitrace_on && trace_syscall_enabled)
  {
    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, SYSCALL_EV, EVT_BEGIN, SYSCALL_SCHED_YIELD_EV);
  }
}

/**
 * Probe_SYSCALL_sched_yield_Exit
 *
 * Probe injected at the end of the syscall 'sched_yield' 
 */
void Probe_SYSCALL_sched_yield_Exit ()
{
  if (mpitrace_on && trace_syscall_enabled)
  {
    TRACE_MISCEVENTANDCOUNTERS(TIME, SYSCALL_EV, EVT_END, SYSCALL_SCHED_YIELD_EV);
  }
}
