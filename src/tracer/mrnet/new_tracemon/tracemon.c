#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <errno.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include "tracemon.h"
#include "common.h"

pthread_t AcceptThread;
int ActiveJobs = 0;
Job_t * JobList = NULL;

static void * TMon_Accept_Connections(void * arg);

int TMon_Initialize()
{
	ActiveJobs = 0;
	JobList = NULL;

	if (pthread_create(&AcceptThread, NULL, TMon_Accept_Connections, NULL) != 0)
	{
		perror("pthread_create");
		exit(1);
	}
	return 1;
}

static void * TMon_Accept_Connections(void * arg)
{
	int sock_fd;
	int port = MONITOR_PORT;
	int cli_len;
	struct sockaddr_in serv_addr, cli_addr;

	sock_fd = socket(AF_INET, SOCK_STREAM, 0);
	if (sock_fd < 0)
	{
		perror("Error opening socket");
		exit(1);
	}

	bzero((char *) &serv_addr, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = INADDR_ANY;
	serv_addr.sin_port = htons(port);

	while (bind(sock_fd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
	{
		if (errno == EADDRINUSE)
		{
			port ++;
			serv_addr.sin_port = htons(port);
		}
		else
		{
			perror("Error on binding");
		}
	}
	fprintf(stderr, "Trace monitor listening for connections on port %d...\n\n", port);

	listen(sock_fd, 5);
	cli_len = sizeof(cli_addr);

	while (1)
	{
		int new_sock_fd;

		new_sock_fd = accept(sock_fd, (struct sockaddr *) &cli_addr, (socklen_t *) &cli_len);
		if (new_sock_fd < 0)
		{
         perror("Error on accept");
		}

		TMon_Register_Job (new_sock_fd);
	}
	return NULL;
}

int TMon_Register_Job (int sock_fd)
{
	int i;

	/* Recycle a previous slot */
	for (i=0; i<ActiveJobs; i++)
	{
		if (JobList[i].sock_fd == -1)
		{
			JobList[i].sock_fd = sock_fd;
			return i;
		}
	}

	/* No slots available, allocate a new one */
	i = ActiveJobs ++;
	JobList = (Job_t *)realloc(JobList, ActiveJobs * sizeof(Job_t));
	JobList[i].sock_fd = sock_fd;
	return i;
}

void TMon_Delete_Job (int job_id)
{
	if ((job_id >= 0) && (job_id < ActiveJobs))
	{
		/* Mark this slot as available */
		JobList[job_id].sock_fd = -1;
		ActiveJobs --;
	}
}

int TMon_Send_Command (int job_id, int command)
{
	if ((job_id >= 0) && (job_id < ActiveJobs))
	{
		return write (JobList[job_id].sock_fd, &command, sizeof(command));
	}
	return -1;
}

int TMon_Read_Data (int job_id, void *buf, int size)
{
    if ((job_id >= 0) && (job_id < ActiveJobs))
	{
		return read (JobList[job_id].sock_fd, buf, size);
	}	
	return -1;
}
