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
		UINT64 n = u * 1000;
		UINT64 d;

		begin = Clock_getCurrentTime(0);
		sleep (u);
		end = Clock_getCurrentTime(0);

		d = (end - begin) / 1000000;

		if ( n != d )
		{
			printf ("sleep(%u) => clocked = %lu nanoseconds (%lu)\n", u, end-begin, d);
			return 1;
		}
	}

	return 0;
}

