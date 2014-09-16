#include <unistd.h>
#include <stdio.h>
#include "common.h"
#include "clock.h"
#include <time.h>

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
		struct timespec ns;
		ns.tv_sec = u;
		ns.tv_nsec = 0;

		begin = Clock_getCurrentTime(0);
		nanosleep (&ns, NULL);
		end = Clock_getCurrentTime(0);

		d = (end - begin) / 1000000;

		if ( u * 1000 != d )
		{
			printf ("nanosleep(%u seconds) => clocked = %lu nanoseconds (%lu)\n", u, end-begin, d);
			return 1;
		}
	}

	return 0;
}

