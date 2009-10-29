/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

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
   
#define MPI_CALLER_EVENT_TYPE(deepness) (Caller_Count[CALLER_MPI] > 1 ? MPI_CALLER_EV+deepness : MPI_CALLER_EV)

enum 
{
	CALLER_MPI = 0,
	CALLER_SAMPLING,
	COUNT_CALLER_TYPES
};

extern int Trace_Caller_Enabled[COUNT_CALLER_TYPES];
extern int *Trace_Caller[COUNT_CALLER_TYPES];
extern int Caller_Deepness[COUNT_CALLER_TYPES];
extern int Caller_Count[COUNT_CALLER_TYPES];

void trace_callers (iotimer_t temps, int deep, int type);
UINT64 get_caller (int deep);

#endif
