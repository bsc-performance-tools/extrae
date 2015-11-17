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

#include <unistd.h>
#include <stdio.h>
#include "common.h"
#include "clock.h"

int main (int argc, char *argv[])
{
	UINT64 begin, end;
	unsigned u;

	UNREFERENCED_PARAMETER(argc);
	UNREFERENCED_PARAMETER(argv);

	Clock_Initialize (1);
	Clock_Initialize_thread ();
	
	for (u = 1 ; u < 5; u++)
	{
		UINT64 d;
		UINT64 n = u * 1000;
		useconds_t us = ((useconds_t) u) * 1000000;

		begin = Clock_getCurrentTime(0);
		usleep (us);
		end = Clock_getCurrentTime(0);

		d = (end - begin) / 1000000;

		/* Allow +- 5 microsecond credit */
		if (!( n-5 <= d && d <= n+5))
		{
			printf ("Executed usleep (%u) but we measured %lu nanoseconds\n",
			  us, end-begin);
			printf ("Comparison of timing in microseconds do not match! (%lu != %lu)\n",
			  n, d);
			return 1;
		}
	}

	return 0;
}

