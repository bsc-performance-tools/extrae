#include <pthread.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

int main (int argc, char *argv[])
{
	int i;
	struct timespec start, end;
	unsigned long long t;

	clock_gettime( CLOCK_REALTIME, &start);
	for (i = 0; i < 10000000; i++)
	{
		pthread_mutex_lock(&lock);
		pthread_mutex_unlock(&lock);
	}
	clock_gettime( CLOCK_REALTIME, &end);

	t = ((end.tv_sec * 1000 * 1000 * 1000) + end.tv_nsec)
	    -((start.tv_sec * 1000 * 1000 * 1000) + start.tv_nsec);

	printf ("time %llu ns aggregated, %llu per loop iteration\n", t, t/10000000);

	return 0;
}

