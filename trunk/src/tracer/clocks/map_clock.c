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
