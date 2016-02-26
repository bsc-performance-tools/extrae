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

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

unsigned long long proc_timebase_MHz = 2000; /* Assume a 2.0GHz machine, in fact, it does not matter */

#if (defined(linux) || defined (__FreeBSD__) || defined (__APPLE__)) && \
    ((__x86_64__) || defined(x86_64) || defined(__amd64__) || defined(amd64) || defined(__i386__))
static __inline unsigned long long ia32_cputime (void)
{
# if defined (__x86_64__) || defined(x86_64) || defined(__amd64__) || defined(amd64)
	unsigned long lo, hi;
# else
	unsigned long long cycles;
# endif

# if defined (__FreeBSD__) || defined (__APPLE__)
	/* 0x0f 0x31 is the bytecode of RDTSC instruction */
#  if defined (__x86_64__) || defined(x86_64) || defined(__amd64__) || defined(amd64)
  /* We cannot use "=A", since this would use %rax on x86_64 */
	__asm __volatile (".byte 0x0f, 0x31" : "=a" (lo), "=d" (hi));
#  else
	__asm __volatile (".byte 0x0f, 0x31" : "=A" (cycles));
# endif
# elif defined(linux)
#  if defined (__x86_64__) || defined(x86_64) || defined(__amd64__) || defined(amd64)
	/* We cannot use "=A", since this would use %rax on x86_64 */
	__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
#  else
	__asm__ __volatile__ ("rdtsc" : "=A" (cycles));
#  endif
# else
#  error "Unknown operating system!"
# endif

# if defined (__x86_64__) || defined(x86_64) || defined(__amd64__) || defined(amd64)
	return ((unsigned long long )hi << 32) | lo;
# else
	return cycles;
# endif
}
#else
static __inline unsigned long long ia32_cputime (void)
{
	return 0;
}
#endif

unsigned long long ia32_getTime (void)
{
  return (ia32_cputime() * 1000) / proc_timebase_MHz; 
}

int main (int argc, char *argv[])
{
	struct timespec start, stop;
	int i;
	unsigned n = 1000000;
	unsigned long long t1, t2, useless;

	clock_gettime (CLOCK_MONOTONIC, &start);
	for (i = 0; i < n; i++)
		useless = ia32_getTime();
	clock_gettime (CLOCK_MONOTONIC, &stop);
	t1 = start.tv_nsec;
	t1 += start.tv_sec * 1000000000ULL;
	t2 = stop.tv_nsec;
	t2 += stop.tv_sec * 1000000000ULL;
	printf ("RESULT : clock_gettime() %Lu ns\n", (t2 - t1) / n);

	return 0;
}
