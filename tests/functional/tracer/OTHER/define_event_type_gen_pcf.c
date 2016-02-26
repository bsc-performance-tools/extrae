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
#include <math.h>

#include "extrae_user_events.h"

double pi_kernel (int n, double h)
{
	double tmp = 0;
	double x;
	int i;

	Extrae_user_function (1);

	for (i = 1; i <= n; i++)
	{
		x = h * ((double)i - 0.5);
		tmp += (4.0 / (1.0 + x*x));
	}

	Extrae_user_function (0);

	return tmp;
}

int main(void)
{
	int n = 1000000;
	double PI25DT = 3.141592653589793238462643;
	double pi, h, area;
	extrae_type_t t1 = 1000, t2 = 2000;
	unsigned nvalues1 = 4, nvalues2 = 0;
	extrae_value_t values[4] = {0, 1, 2, 3};
	char * description_values[4] = {"End", "Begin", "Phase1", "Phase2" };

	Extrae_init();
	Extrae_define_event_type (&t1, "Kernel execution", &nvalues1, values, description_values);
	Extrae_define_event_type (&t2, "Kernel execution_2", &nvalues2, NULL, NULL);

	h = 1.0 / (double) n;

	Extrae_event (1000, 1);
	area = pi_kernel (n, h);
	Extrae_event (1000, 0);
	pi = h * area;

	printf("pi is approximately %.16f, Error is %.16f\n",pi,fabs(pi - PI25DT));

	Extrae_fini();
	return 0;
}
