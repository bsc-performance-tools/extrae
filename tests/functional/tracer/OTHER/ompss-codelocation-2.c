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
	extrae_combined_events_t evt;

	extrae_type_t type = 2020;
	extrae_value_t enter = (extrae_value_t) pi_kernel, leave = 0;

	Extrae_register_function_address (pi_kernel, (char*)__FUNCTION__, (char*)__FILE__, __LINE__);

	Extrae_init_CombinedEvents (&evt);
	evt.nEvents = 1;
	evt.Types = &type;
	evt.Values = &enter;
	Extrae_emit_CombinedEvents (&evt);

	for (i = 1; i <= n; i++)
	{
		x = h * ((double)i - 0.5);
		tmp += (4.0 / (1.0 + x*x));
	}

	evt.nEvents = 1;
	evt.Types = &type;
	evt.Values = &leave;
	Extrae_emit_CombinedEvents (&evt);

	return tmp;
}

void sleep_kernel (int n)
{
	extrae_combined_events_t evt;
	extrae_type_t type = 2020;
	extrae_value_t enter = (extrae_value_t) sleep_kernel, leave = 0;

	Extrae_register_function_address (sleep_kernel, (char*)__FUNCTION__, (char*)__FILE__, __LINE__);

	Extrae_init_CombinedEvents (&evt);
	evt.nEvents = 1;
	evt.Types = &type;
	evt.Values = &enter;
	Extrae_emit_CombinedEvents (&evt);

	printf ("in sleep_kernel (%d)\n", n);

	evt.nEvents = 1;
	evt.Types = &type;
	evt.Values = &leave;
	Extrae_emit_CombinedEvents (&evt);
}


void fake_kernel (void)
{
	printf ("in fake_kernel ()\n");
}

void fake_kernel_ol_1 (void)
{
	Extrae_register_function_address (fake_kernel_ol_1, "fake_kernel", (char*)__FILE__, __LINE__);

	extrae_combined_events_t evt;
	extrae_type_t type = 2020;
	extrae_value_t enter = (extrae_value_t) fake_kernel_ol_1, leave = 0;

	Extrae_init_CombinedEvents (&evt);
	evt.nEvents = 1;
	evt.Types = &type;
	evt.Values = &enter;
	Extrae_emit_CombinedEvents (&evt);

	fake_kernel();

	evt.nEvents = 1;
	evt.Types = &type;
	evt.Values = &leave;
	Extrae_emit_CombinedEvents (&evt);
}

void fake_kernel_ol_2 (void)
{
	Extrae_register_function_address (fake_kernel_ol_2, "fake_kernel", (char*)__FILE__, __LINE__);

	extrae_combined_events_t evt;
	extrae_type_t type = 2020;
	extrae_value_t enter = (extrae_value_t) fake_kernel_ol_2, leave = 0;

	Extrae_init_CombinedEvents (&evt);
	evt.nEvents = 1;
	evt.Types = &type;
	evt.Values = &enter;
	Extrae_emit_CombinedEvents (&evt);

	fake_kernel();

	evt.nEvents = 1;
	evt.Types = &type;
	evt.Values = &leave;
	Extrae_emit_CombinedEvents (&evt);
}

int main (void)
{
	int n = 1000;
	double PI25DT = 3.141592653589793238462643;
	double pi, h, area;

	Extrae_register_codelocation_type (2000, 2020, "Function-F", "Line-Source-F");

	h = 1.0 / (double) n;

	area = pi_kernel (n, h);
	pi = h * area;
	printf("pi is approximately %.16f, Error is %.16f\n",pi,fabs(pi - PI25DT));

	sleep_kernel (2);

	fake_kernel_ol_1 ();
	fake_kernel_ol_2 ();

	return 0;
}
