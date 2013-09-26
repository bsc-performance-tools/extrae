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
		useconds_t us = ((useconds_t) u) * 1000000;

		begin = Clock_getCurrentTime(0);
		usleep (us);
		end = Clock_getCurrentTime(0);

		d = (end - begin) / 1000000;

		if ( u * 1000 != d )
		{
			printf ("usleep(%u) => clocked = %lu nanoseconds (%lu)\n", us, end-begin, d);
			return 1;
		}
	}

	return 0;
}

