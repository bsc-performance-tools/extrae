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

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_TIME_H
# include <time.h>
#endif

#include <clock.h>

#if _POSIX_C_SOURCE < 199309L
#ifndef OS_ANDROID
# error "Looks like _POSIX_C_SOURCE is not appropriate to compile this file!"
#endif
#endif

#include "posix_clock.h"

void posix_Initialize (void)
{
	/* Do nothing */
}

void posix_Initialize_thread (void)
{
	/* Do nothing */
}

iotimer_t posix_getTime (void)
{
	iotimer_t t_sec, t_nsec;
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);

	t_sec = ts.tv_sec;
	t_nsec = ts.tv_nsec;
	return t_sec * 1000000000 + t_nsec;
}

