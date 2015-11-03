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

