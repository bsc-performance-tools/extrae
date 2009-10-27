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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/clocks/ia32_clock.c,v $
 | 
 | @last_commit: $Date: 2008/09/02 16:19:15 $
 | @version:     $Revision: 1.12 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: ia32_clock.c,v 1.12 2008/09/02 16:19:15 harald Exp $";

#if (defined(OS_LINUX) || defined(OS_FREEBSD) || defined(OS_SOLARIS)) && defined(ARCH_IA32)

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#if defined(OS_LINUX)
# ifdef HAVE_STRING_H
#  include <string.h>
# endif
#elif defined(OS_FREEBSD)
# ifdef HAVE_SYS_TYPES_H
#  include <sys/types.h>
# endif
# ifdef HAVE_SYS_SYSCTL_H
#  include <sys/sysctl.h>
# endif
# ifdef HAVE_STDLIB_H
#  include <stdlib.h>
# endif
#elif defined(OS_SOLARIS)
# ifdef HAVE_SYS_TIME_H
#  include <sys/time.h>
# endif
#endif

#include "ia32_clock.h"

static unsigned long long proc_timebase_MHz;

iotimer_t ia32_getTime (void);
void ia32_Initialize (void);
void ia32_Initialize_thread (void);

#if defined(OS_FREEBSD) || defined (OS_LINUX)
static __inline unsigned long long ia32_cputime (void)
{
#if defined(ARCH_IA32_x64)
	unsigned long lo, hi;
#else
	unsigned long long cycles;
#endif

#if defined(OS_FREEBSD)
	/* 0x0f 0x31 is the bytecode of RDTSC instruction */
# if defined (ARCH_IA32_x64)
  /* We cannot use "=A", since this would use %rax on x86_64 */
	__asm __volatile (".byte 0x0f, 0x31" : "=a" (lo), "=d" (hi));
# else
	__asm __volatile (".byte 0x0f, 0x31" : "=A" (cycles));
#endif
#elif defined(OS_LINUX)
# if defined (ARCH_IA32_x64)
	/* We cannot use "=A", since this would use %rax on x86_64 */
	__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
# else
	__asm__ __volatile__ ("rdtsc" : "=A" (cycles));
# endif
#else
# error "Unknown operating system!"
#endif

#if defined (ARCH_IA32_x64)
	return ((unsigned long long )hi << 32) | lo;
#else
	return cycles;
#endif
}

void ia32_Initialize (void)
{
#if defined(OS_LINUX)
  FILE *fp;
  char buffer[ 32768 ];
  size_t bytes_read;
  char* match;
  double temp;
  int res;

  fp = fopen( "/proc/cpuinfo", "r" );
  bytes_read = fread( buffer, 1, sizeof( buffer ), fp );
  fclose( fp );

  if (bytes_read == 0)
    return;

  buffer[ bytes_read ] = '\0';
  match = strstr( buffer, "cpu MHz" );
  if (match == NULL)
    return;

  res = sscanf (match, "cpu MHz    : %lf", &temp );

  proc_timebase_MHz = (res == 1) ? temp : 0;
#elif defined(OS_FREEBSD)
	int mib[3];
	int result;
	unsigned len = 3;
	unsigned long long tsc_value;

  result = sysctlnametomib ("machdep.tsc_freq", mib, &len);
  if (result == -1)
  {
    perror ("sysctlnametomib");
    exit (-1);
  }

  len = sizeof(tsc_value);
  result = sysctl (mib, 2, &tsc_value, &len, NULL, 0);
  if (result == -1)
  {
    perror ("sysctl");
    exit (-1);
  }

  proc_timebase_MHz = (unsigned long long) (tsc_value / 1000000);
#endif
}
#endif /* OS_FREEBSD || OS_LINUX */

void ia32_Initialize_thread (void)
{
	/* Do nothing */
}

iotimer_t ia32_getTime (void)
{
#if defined(OS_FREEBSD) || defined(OS_LINUX)
  return (ia32_cputime() * 1000) / proc_timebase_MHz; 
#elif defined (OS_SOLARIS)
  return gethrtime();
#endif
}

#endif
