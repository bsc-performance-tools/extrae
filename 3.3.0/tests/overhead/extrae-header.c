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

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_TIME_H
# include <time.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif

#ifdef HAVE_LIBUNWIND_H
# include <libunwind.h>
#endif

#ifdef HAVE_PAPI_H
# include  <papi.h>
#endif

static char * search_in_cpu_info (char *search_str)
{
	char *s;
	char line[1024];

	FILE *f = fopen ("/proc/cpuinfo", "r");
	if (f != NULL)
	{
		while ( fgets( line, 256, f ) != NULL ) {
			if ( strstr( line, search_str ) != NULL ) {
				for ( s = line; *s && ( *s != ':' ); ++s );
				if ( *s )
					return s;
			}
		}
	}
	return NULL;
}

int main (int argc, char *argv[])
{
	UNREFERENCED_PARAMETER(argc);
	UNREFERENCED_PARAMETER(argv);
	char *cpuinfo;

	printf (PACKAGE_STRING" SVN revision %d based on " EXTRAE_SVN_BRANCH"\n", EXTRAE_SVN_REVISION);
	cpuinfo = search_in_cpu_info ("model name");
	if (cpuinfo == NULL)
		cpuinfo = search_in_cpu_info ("Processor");
	printf ("CPU info%s", cpuinfo);

#if defined(HAVE_LIBUNWIND_H)
    printf ("Using libunwind v%d.%d from %s\n",
	  UNW_VERSION_MAJOR,
	  UNW_VERSION_MINOR,
	  UNWIND_HOME);
#else
	printf ("Using backtrace or don't using calltrace?\n");
#endif

#if defined(HAVE_PAPI_H)
	printf ("Using PAPI v%d.%d.%d from %s\n",
	  PAPI_VERSION_MAJOR(PAPI_VERSION),
	  PAPI_VERSION_MINOR(PAPI_VERSION),
	  PAPI_VERSION_REVISION(PAPI_VERSION),
	  PAPI_HOME);
#elif defined(PMAPI_COUNTERS)
	printf ("Using PMAPI\n");
#else
	printf ("Don't using hwc\n");
#endif

	printf ("Running on: ");
#if defined(linux)
	printf ("linux");
#elif defined(__FreeBSD__)
	printf ("FreeBSD");
#elif defined(__APPLE__)
	printf ("MacOS");
#elif defined(_AIX)
	printf ("AIX");
#else
	printf ("unknown OS");
#endif
	printf (" running on ");
#if defined(__x86_64__) || defined(x86_64) || defined(__amd64__) || defined(amd64)
	printf ("x86 with 64 bit extensions");
#elif defined(__i386__)
	printf ("x86");
#elif defined(__ia64__)
	printf ("intel itanium");
#elif defined(__powerpc__)
	printf ("powerpc");
#elif defined(__powerpc64__)
	printf ("powerpc with 64 bit extensions");
#elif defined(__arm__)
	printf ("ARM");
#elif defined(__aarch64__)
	printf ("ARM64");
#else
	printf ("unknown");
#endif
	printf (" processor\n");

	return 0;
}
