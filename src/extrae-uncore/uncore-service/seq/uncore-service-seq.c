#include <unistd.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

//#define DEBUG

#if defined(DEBUG)
# define debug_msg(stream, fmt, ...) fprintf(stream, fmt, ##__VA_ARGS__)
#else
# define debug_msg(stream, fmt, ...)
#endif

int stop_signal = 0;

void exit_handler(int signum)
{
	if (signum == SIGQUIT)
	{
		stop_signal = 1;
	}
}


int main(int argc, char **argv)
{
	debug_msg(stderr, "[DEBUG] UNCORE SERVICE INITIATES\n");

	signal(SIGQUIT, exit_handler);

	while(!stop_signal) sleep(1);

	debug_msg(stderr, "[DEBUG] UNCORE READER QUITS\n");

	return 0;
}
