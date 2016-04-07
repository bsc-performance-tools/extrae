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

#include "wrapper.h"
#include "trace_macros.h"
#include "io_probe.h"

#if defined (INSTRUMENT_IO)

//#define DEBUG

/*
	This wrappers can only be compiled if the file is being compiled to
	generate a shared library (-DPIC)
*/

static ssize_t (*real_read)(int fd, void *buf, size_t count) = NULL;
static ssize_t (*real_write)(int fd, const void *buf, size_t count) = NULL;

void Extrae_iotrace_init (void)
{
# if defined(PIC) /* This is only available through .so libraries */
	real_read = (ssize_t(*)(int, void*, size_t)) dlsym (RTLD_NEXT, "read");
	real_write = (ssize_t(*)(int, const void*, size_t)) dlsym (RTLD_NEXT, "write");
# else
	fprintf (stderr, PACKAGE_NAME": Warning! I/O instrumentation requires linking with shared library!\n");
# endif
}

/*

   INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
   INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE

*/

#define TRACE_IO_CALLER_IS_ENABLED \
 (Trace_Caller_Enabled[CALLER_IO])

#define TRACE_IO_CALLER(evttime,offset) \
{ \
	if (TRACE_IO_CALLER_IS_ENABLED) \
		Extrae_trace_callers (evttime, offset, CALLER_IO); \
}



# if defined(PIC) /* This is only available through .so libraries */
ssize_t read (int f, void *b, size_t s)
{
	int canInstrument = !Backend_inInstrumentation(THREADID) && 
	  mpitrace_on &&
	  Extrae_get_trace_io();
	ssize_t res;

	if (real_read == NULL)
		Extrae_iotrace_init();

#if defined(DEBUG)
	if (canInstrument)
	{
		fprintf (stderr, PACKAGE_NAME": read is at %p\n", real_read);
		fprintf (stderr, PACKAGE_NAME": read params %d %p %lu\n", f, b, s);
	}
#endif

	if (real_read != NULL && canInstrument)
	{
		Backend_Enter_Instrumentation (2);
		Probe_IO_read_Entry (f, s);
		TRACE_IO_CALLER(LAST_READ_TIME, 3);
		res = real_read (f, b, s);
		Probe_IO_read_Exit ();
		Backend_Leave_Instrumentation ();
	}
	else if (real_read != NULL && !canInstrument)
	{
		res = real_read (f, b, s);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": read is not hooked! exiting!!\n");
		abort();
	}

    return res;
}

ssize_t write (int f, const void *b, size_t s)
{
	int canInstrument = !Backend_inInstrumentation(THREADID) && 
	  mpitrace_on &&
	  Extrae_get_trace_io();
	ssize_t res;

	if (real_write == NULL)
		Extrae_iotrace_init();

#if defined(DEBUG)
	if (canInstrument)
	{
		fprintf (stderr, PACKAGE_NAME": write is at %p\n", real_write);
		fprintf (stderr, PACKAGE_NAME": write params %d %p %lu\n", f, b, s);
	}
#endif

	if (real_write != NULL && canInstrument)
	{
		Backend_Enter_Instrumentation (2);
		Probe_IO_write_Entry (f, s);
		TRACE_IO_CALLER(LAST_READ_TIME, 3);
		res = real_write (f, b, s);
		Probe_IO_write_Exit ();
		Backend_Leave_Instrumentation ();
	}
	else if (real_write != NULL && !canInstrument)
	{
		res = real_write (f, b, s);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": write is not hooked! exiting!!\n");
		abort();
	}

    return res;
}
# endif /* -DPIC */

#endif /* INSTRUMENT_IO */
