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
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#include "map_clock.h"

#if defined(IS_BGL_MACHINE)
# include "bgl_clock.h"
#elif (defined(OS_LINUX) || defined(OS_AIX)) && defined(ARCH_PPC)
# include "ppc_clock.h"
#elif defined (OS_LINUX) && defined (ARCH_IA64)
# include "ia64_clock.h"
#elif (defined (OS_LINUX) || defined(OS_FREEBSD) || defined(OS_SOLARIS)) && defined(ARCH_IA32)
# include "ia32_clock.h"
#else
# ifdef HAVE_SYS_TIME_H
#  include <sys/time.h>
# endif
#endif


/* C interface */

#if defined(DEAD_CODE)
unsigned int map_clock (void)
{
  return 0;
}
#endif

iotimer_t get_hr_timer (void)
{
#if defined(IS_BGL_MACHINE)
  return getBGLtime ();
#elif (defined(OS_LINUX) || defined(OS_AIX)) && defined(ARCH_PPC)
  return getPPCtime ();
#elif defined (OS_LINUX) && defined (ARCH_IA64)
  return getIA64time ();
#elif (defined (OS_LINUX) || defined(OS_FREEBSD) || defined(OS_SOLARIS)) && defined (ARCH_IA32)
  return getIA32time ();
#else
  struct timeval aux;
  gettimeofday (&aux, NULL);
  return (((iotimer_t) aux.tv_sec) * 1000000 + aux.tv_usec);
#endif
}

#if defined(DEAD_CODE)
double hr_to_secs (iotimer_t hr)
{
  return (((double) hr) / 1000000);
}

double hr_to_ms (iotimer_t hr)
{
  return (((double) hr) / 1000);
}

double hr_to_us (iotimer_t hr)
{
  return ((double) hr);
}
#endif /* DEAD_CODE */

#if defined(DEAD_CODE)

/* FORTRAN interface */
#ifndef PMPI_NO_UNDERSCORES

unsigned int CtoF77 (map_clock) (void)
{
  return (0);
}

iotimer_t CtoF77 (get_hr_timer) (void)
{
#if defined(IS_BGL_MACHINE)
  return getBGLtime ();
#elif (defined(OS_LINUX) || defined(OS_AIX)) && defined(ARCH_PPC)
  return getPPCtime ();
#elif defined(OS_LINUX) && defined(ARCH_IA64)
  return getIA64time ();
#elif (defined (OS_LINUX) || defined(OS_FREEBSD) || defined(OS_SOLARIS)) && defined (ARCH_IA32)
  return getIA32time ();
#else
  struct timeval aux;
  gettimeofday (&aux, NULL);
  return (((iotimer_t) aux.tv_sec) * 1000000 + aux.tv_usec);
#endif
}

double CtoF77 (hr_to_secs) (iotimer_t * hr)
{
  return (((double) *hr) / 1000000);
}

double CtoF77 (hr_to_ms) (iotimer_t * hr)
{
  return (((double) *hr) / 1000);
}

double CtoF77 (hr_to_us) (iotimer_t * hr)
{
  return ((double) *hr);
}
#endif

#endif /* DEAD_CODE */
