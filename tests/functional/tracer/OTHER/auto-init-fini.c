#include "extrae_user_events.h"

#define UNREFERENCED(x) ((x)=(x))

int main (int argc, char *argv[])
{
	UNREFERENCED(argc);
	UNREFERENCED(argv);

	/* This should be automatically called */
	/* Extrae_init(); */

	Extrae_event (1234, 1);
	Extrae_event (1234, 0);

	/* This should be automatically called */
	/* Extrae_fini(); */

	return 0;
}
