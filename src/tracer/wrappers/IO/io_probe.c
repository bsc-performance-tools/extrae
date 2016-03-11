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

#include "threadid.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "io_probe.h"

static int trace_io = FALSE;

void Extrae_set_trace_io (int b)
{ trace_io = b; }

int Extrae_get_trace_io (void)
{ return trace_io; }

static unsigned Extrae_get_descriptor_type (int fd)
{
	if (isatty(fd))
	{
		return DESCRIPTOR_TYPE_ATTY;
	}
	else
	{
		struct stat buf;
		fstat (fd, &buf);
		if (S_ISREG(buf.st_mode))
			return DESCRIPTOR_TYPE_REGULARFILE;
		else if (S_ISSOCK(buf.st_mode))
			return DESCRIPTOR_TYPE_SOCKET;
		else if (S_ISFIFO(buf.st_mode))
			return DESCRIPTOR_TYPE_FIFO_PIPE;
		else
			return DESCRIPTOR_TYPE_UNKNOWN;
	}
}

void Probe_IO_write_Entry (int f, size_t s)
{
	if (mpitrace_on && trace_io)
	{
		unsigned type = Extrae_get_descriptor_type (f);
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, WRITE_EV, EVT_BEGIN, f);
		TRACE_MISCEVENT(LAST_READ_TIME, WRITE_EV, EVT_BEGIN+1, s);
		TRACE_MISCEVENT(LAST_READ_TIME, WRITE_EV, EVT_BEGIN+2, type);
	}
}

void Probe_IO_write_Exit (void)
{
	if (mpitrace_on && trace_io)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, WRITE_EV, EVT_END, EMPTY);
	}
}

void Probe_IO_read_Entry (int f, size_t s)
{
	if (mpitrace_on && trace_io)
	{
		unsigned type = Extrae_get_descriptor_type (f);
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, READ_EV, EVT_BEGIN, f);
		TRACE_MISCEVENT(LAST_READ_TIME, READ_EV, EVT_BEGIN+1, s);
		TRACE_MISCEVENT(LAST_READ_TIME, READ_EV, EVT_BEGIN+2, type);
	}
}

void Probe_IO_read_Exit (void)
{
	if (mpitrace_on && trace_io)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, READ_EV, EVT_END, EMPTY);
	}
}
