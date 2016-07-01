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

#if (defined(OS_LINUX) || defined(OS_FREEBSD) || defined (OS_DARWIN) || defined(OS_SOLARIS)) && defined(ARCH_IA32)

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#if defined(OS_LINUX)
# ifdef HAVE_STRING_H
#  include <string.h>
# endif
#elif defined(OS_FREEBSD) || defined(OS_DARWIN)
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

#if defined(OS_FREEBSD) || defined (OS_LINUX) || defined (OS_DARWIN)
static __inline unsigned long long ia32_cputime (void)
{
#if defined(ARCH_IA32_x64)
	unsigned long lo, hi;
#else
	unsigned long long cycles;
#endif

#if defined(OS_FREEBSD) || defined (OS_DARWIN)
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
#elif defined(OS_FREEBSD) || defined(OS_DARWIN)
	int mib[3];
	int result;
	size_t len = 2;
	unsigned long long tsc_value;

#if defined(OS_FREEBSD)
  result = sysctlnametomib ("machdep.tsc_freq", mib, &len);
#elif defined (OS_DARWIN)
  result = sysctlnametomib ("hw.cpufrequency", mib, &len);
#endif
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
#endif /* OS_FREEBSD || OS_LINUX || OS_DARWIN */

iotimer_t ia32_getTime (void)
{
#if defined(OS_FREEBSD) || defined(OS_LINUX) || defined(OS_DARWIN)
  return (ia32_cputime() * 1000) / proc_timebase_MHz; 
#elif defined (OS_SOLARIS)
  return gethrtime();
#endif
}

#endif
