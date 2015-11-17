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

#include "extrae_user_events.h"

#define UNREFERENCED(x) ((x)=(x))

#define N 16

int main (int argc, char *argv[])
{
	unsigned u;
	extrae_combined_events_t evt;
	extrae_type_t type = 1234;
	extrae_value_t enter = 1, leave = 0;
	extrae_type_t multiple_types[N];
	extrae_value_t multiple_values[N];

	UNREFERENCED(argc);
	UNREFERENCED(argv);

	Extrae_init();

	/* Check combined events */

	Extrae_init_CombinedEvents (&evt);
	evt.nEvents = 1;
	evt.Types = &type;
	evt.Values = &enter;
	Extrae_emit_CombinedEvents (&evt);
	evt.nEvents = 1;
	evt.Types = &type;
	evt.Values = &leave;
	Extrae_emit_CombinedEvents (&evt);

	/* Check regular events */

	Extrae_event (4321, 1);
	Extrae_event (4321, 0);

	/* Check regular multiple events */
	for (u = 0; u < N; u++)
	{
		multiple_types[u] = u;
		multiple_values[u] = u+1;
	}
	Extrae_nevent (N, multiple_types, multiple_values);

	Extrae_fini();

	return 0;
}
