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

#ifdef HAVE_SYS_RESOURCE_H
# include <sys/resource.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif

#include "cpu_clock.h"

iotimer_t get_cpu_time (void)
{
  struct rusage aux;
  iotimer_t temps;

  if (getrusage (RUSAGE_SELF, &aux) < 0)
  {
    fprintf (stderr, "EL GETRUSAGE HA DONAT ERROR!\n");
    return (0);
  }
  temps = ((iotimer_t) aux.ru_utime.tv_sec) * 1000000 + aux.ru_utime.tv_usec;
  temps += ((iotimer_t) aux.ru_stime.tv_sec) * 1000000 + aux.ru_stime.tv_usec;

  return (temps);
}



double cpu_time_to_sec (iotimer_t value)
{
  double res;

  res = (((double) value) / 1000000);
  return (res);
}


/* FORTRAN interface */

iotimer_t get_cpu_time_ (void)
{
  // Retorna el temps de CPU utilitzat pel proces (en microsegons)
  struct rusage aux;
  iotimer_t temps;

  if (getrusage (RUSAGE_SELF, &aux) < 0)
    return (0);
  temps = ((iotimer_t) aux.ru_utime.tv_sec) * 1000000 + aux.ru_utime.tv_usec;
  temps += ((iotimer_t) aux.ru_stime.tv_sec) * 1000000 + aux.ru_stime.tv_usec;

  return (temps);
}


double cpu_time_to_sec_ (iotimer_t value)
{
  return (((double) value) / 1000000);
}
