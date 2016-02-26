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

#ifndef CLOCK_H
#define CLOCK_H

#include "threadid.h"

enum
{
  REAL_CLOCK = 0,
  USER_CLOCK
};

#if SIZEOF_LONG_LONG == 8 
typedef unsigned long long iotimer_t;
#elif SIZEOF_LONG == 8
typedef unsigned long iotimer_t;
#endif

#define TIME (Clock_getCurrentTime(THREADID))
#define LAST_READ_TIME (Clock_getLastReadTime(THREADID))
#define CLOCK_INIT(x) (Clock_Initialize(x))
#define CLOCK_INIT_THREAD (Clock_Initialize())

#if defined(__cplusplus)
extern "C" {
#endif
void Clock_setType (unsigned type);
unsigned Clock_getType (void);

UINT64 Clock_getCurrentTime_nstore (); 
UINT64 Clock_getCurrentTime (unsigned thread);
UINT64 Clock_getLastReadTime (unsigned thread);
void Clock_AllocateThreads (unsigned numthreads);
void Clock_Initialize (unsigned numthreads);
void Clock_Initialize_thread (void);
void Clock_CleanUp (void);
#if defined(__cplusplus)
}
#endif

#endif
