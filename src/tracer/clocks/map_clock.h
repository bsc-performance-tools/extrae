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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/clocks/map_clock.h,v $
 | 
 | @last_commit: $Date: 2007/09/21 16:33:40 $
 | @version:     $Revision: 1.1.1.1 $
 | 
 | History:
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
