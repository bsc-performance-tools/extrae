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
