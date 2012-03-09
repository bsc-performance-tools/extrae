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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include <stdio.h>
#include <omp.h>
#include <math.h>

#include "extrae_user_events.h"

int main(int argc, char **argv)
{
	int i;
	int n = 1000000;
	double PI25DT = 3.141592653589793238462643;
	double pi, h, area, x;

	/* Extrae_init() must be called before any #pragma omp or OMP call */
	Extrae_init();

	h = 1.0 / (double) n;
	area = 0.0;
	#pragma omp parallel for private(x) reduction(+:area)
	for (i = 1; i <= n; i++)
	{
		x = h * ((double)i - 0.5);
		area += (4.0 / (1.0 + x*x));
	}
	pi = h * area;
	printf("pi (by using #pragma omp parallel for) is approximately %.16f, Error is %.16f\n",pi,fabs(pi - PI25DT));

	h = 1.0 / (double) n;
	area = 0.0;
	#pragma omp parallel sections private(i,x) reduction(+:area)
	{
		#pragma omp section
		for (i = 1; i < n/2; i++)
		{
			x = h * ((double)i - 0.5);
			area += (4.0 / (1.0 + x*x));
		}

		#pragma omp section
		for (i = n/2; i <= n; i++)
		{
			x = h * ((double)i - 0.5);
			area += (4.0 / (1.0 + x*x));
		}
	}
	pi = h * area;
	printf("pi (by using #pragma omp parallel sections) is approximately %.16f, Error is %.16f\n",pi,fabs(pi - PI25DT));


	/* Extre_fini() must be the last call */
	Extrae_fini();
}
