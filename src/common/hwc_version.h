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

#ifndef __HWC_VERSION__
#define __HWC_VERSION__

#include <config.h>

#if defined(HETEROGENEOUS_SUPPORT)
# define PAPIv3
#else
# if USE_HARDWARE_COUNTERS
#  if defined(PAPI_COUNTERS)
#   include <papi.h>
#   if PAPI_VER_CURRENT == 2 /* PAPI 2.x */
#    define PAPIv2
#   elif PAPI_VERSION_MAJOR(PAPI_VERSION) == 3 /* PAPI 3.x */
#    define PAPIv3
#   endif
#  elif defined(PMAPI_COUNTERS)
#   include <pmapi.h>
#  endif
# endif
#endif

#endif /* __HWC_VERSION__ */

