#include <pthread.h>
#include <stdio.h>

void *pthread_func(void *useless)
{
	useless = useless;
	return NULL;
}

int main()
{
	pthread_t pt;
	
	if (pthread_create (&pt, NULL, pthread_func, NULL))
		return 1;
	
	if (pthread_join (pt, NULL))
		return 2;

	return 0;
}
