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

#ifndef _CALLTRACE_H_INCLUDED_
#define _CALLTRACE_H_INCLUDED_

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_ERRNO_H
# include <errno.h>
#endif
#ifdef HAVE_LIMITS_H
# include <limits.h>
#endif

#include "common.h"
#include "clock.h"

/*
#define MAX_CALLERS            100
*/
#define FOUR_CALLS_AGO           4
#define FIVE_CALLS_AGO           5
#define MAX_USER_FUNCTION_OFFSET FIVE_CALLS_AGO
#define MAX_STACK_DEEPNESS       (MAX_CALLERS + MAX_USER_FUNCTION_OFFSET)
   
#define MPI_CALLER_EVENT_TYPE(deepness) (Caller_Count[CALLER_MPI] >= 1 ? CALLER_EV+deepness : CALLER_EV)

#define CALLER_EVENT_TYPE(type,deepness) \
	(Caller_Count[type] >= 1 ? CALLER_EV+deepness : CALLER_EV)

enum 
{
	CALLER_MPI = 0,
	CALLER_SAMPLING,
	CALLER_DYNAMIC_MEMORY,
	CALLER_IO,
	COUNT_CALLER_TYPES
};

extern int Trace_Caller_Enabled[COUNT_CALLER_TYPES];
extern int *Trace_Caller[COUNT_CALLER_TYPES];
extern int Caller_Deepness[COUNT_CALLER_TYPES];
extern int Caller_Count[COUNT_CALLER_TYPES];

void Extrae_trace_callers (iotimer_t temps, int deep, int type);
UINT64 Extrae_get_caller (int deep);

#endif
