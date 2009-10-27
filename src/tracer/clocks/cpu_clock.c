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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/clocks/cpu_clock.c,v $
 | 
 | @last_commit: $Date: 2008/01/26 11:18:22 $
 | @version:     $Revision: 1.2 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: cpu_clock.c,v 1.2 2008/01/26 11:18:22 harald Exp $";

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
