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

#if HAVE_STDIO_H
# include <stdio.h>
#endif
#if HAVE_DLFCN_H
# define __USE_GNU
# include <dlfcn.h>
# undef __USE_GNU
#endif
#if HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_SYS_UIO_H
# include <sys/uio.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_STDARG_H
# include <stdarg.h>
#endif

#include "syscall_wrapper.h"
#include "syscall_probe.h"
#include "wrapper.h"

#if defined(INSTRUMENT_SYSCALL)

//#define DEBUG

/***************************************************************************************\
 * This file contains wrappers to instrument system calls other than I/O and MALLOCS.  *
 * The interposition of these wrappers require a shared library (-DPIC).               *
\***************************************************************************************/

/* Global pointers to the real implementation of the system calls */
static int (*real_sched_yield)(void) = NULL;

/**
 * Extrae_iotrace_init
 *
 * Initialization routine for the I/O tracing module. Performs a discovery of the
 * address of the real implementation of the I/O calls through dlsym. The initialization
 * is deferred until any of the instrumented symbols is used for the first time.
 */
void Extrae_syscalltrace_init (void)
{
# if defined(PIC) /* Only available for .so libraries */

  /*
   * Find the first implementation of the system call in the default library search order
   * after the current library. Not finding any of the symbols doesn't throw an error
   * unless the application tries to use it later.
   */
  real_sched_yield = (int(*)(void)) dlsym(RTLD_NEXT, "sched_yield");

#  if defined(DEBUG)
  fprintf(stderr, "[DEBUG] Extrae installed the following system call hooks:\n");
  fprintf(stderr, "[DEBUG] sched_yield hooked at %p\n", real_sched_yield);
#  endif

# else

  fprintf (stderr, PACKAGE_NAME": Warning! System call instrumentation requires linking with shared library!\n");

# endif
}

# if defined(PIC) /* Only available for .so libraries */

/**
 * sched_yield
 *
 * Wrapper for the system call 'sched_yield'
 */
int sched_yield(void)
{
	int rc;

  /* Check whether SYSCALL instrumentation is enabled */
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_syscall();

  /* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  /* Initialize the module if the pointer to the real call is not yet set */
  if (real_sched_yield == NULL)
  {
    Extrae_syscalltrace_init();
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf (stderr, PACKAGE_NAME": sched_yield is at %p\n", real_sched_yield);
  }
#endif

  if (real_sched_yield != NULL && canInstrument)
  {
    /* Instrumentation is enabled, emit events and invoke the real call */
    Backend_Enter_Instrumentation (2+Caller_Count[CALLER_SYSCALL]);
    Probe_SYSCALL_sched_yield_Entry ();
    TRACE_SYSCALL_CALLER(LAST_READ_TIME, 3);

		rc = real_sched_yield();
    Probe_SYSCALL_sched_yield_Exit ();
    Backend_Leave_Instrumentation ();
  }
  else if (real_sched_yield != NULL && !canInstrument)
  {
    /* Instrumentation is not enabled, bypass to the real call */
    rc = real_sched_yield();
  }
  else
  {
    /*
     * An error is thrown if the application uses this symbol but during the initialization
     * we couldn't find the real implementation. This kind of situation could happen in the
     * very strange case where, by the time this symbol is first called, the libc (where the
     * real implementation is) has not been loaded yet. One suggestion if we see this error
     * is to try to prepend the libc.so to the LD_PRELOAD to force to load it first.
     */
    fprintf (stderr, PACKAGE_NAME": sched_yield is not hooked! exiting!!\n");
    abort();
  }
  return rc;
}

# endif /* -DPIC */

#endif /* -DINSTRUMENT_SYSCALL */

