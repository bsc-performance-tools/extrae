/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef _MAP_CLOCK_H
#define _MAP_CLOCK_H

#include <config.h>

#include "defines.h"            /* Per la macro CtoF77 */
#include "common.h"

#if SIZEOF_LONG == 8 
typedef unsigned long iotimer_t;
#elif SIZEOF_LONG_LONG == 8
typedef unsigned long long iotimer_t;
#endif

/* C interface */

unsigned int map_clock ();
iotimer_t get_hr_timer ();
double hr_to_secs (iotimer_t hr);
double hr_to_ms (iotimer_t hr);
double hr_to_us (iotimer_t hr);

#ifndef PMPI_NO_UNDERSCORES

unsigned int CtoF77 (map_clock) ();
iotimer_t CtoF77 (get_hr_timer) ();
double CtoF77 (hr_to_ms) (iotimer_t * hr);
double CtoF77 (hr_to_us) (iotimer_t * hr);
double CtoF77 (hr_to_secs) (iotimer_t * hr);

#endif

#endif
