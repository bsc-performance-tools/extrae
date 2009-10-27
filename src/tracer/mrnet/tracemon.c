/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/mrnet/tracemon.c,v $
 | 
 | @last_commit: $Date: 2008/12/01 10:39:14 $
 | @version:     $Revision: 1.3 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: tracemon.c,v 1.3 2008/12/01 10:39:14 gllort Exp $";

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
# include <sys/socket.h>
#endif
#ifdef HAVE_NETINET_IN_H
# include <netinet/in.h>
#endif
#ifdef HAVE_NETDB_H
# include <netdb.h>
#endif
#ifdef HAVE_ERRNO_H
# include <errno.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_PTHREAD_H
# include <pthread.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif

#define MONITOR_PORT 12000

typedef struct 
{
   int sockfd;
} 
job_t;

typedef struct
{
   job_t ** jobs_list;
   int count;
}
running_jobs_t;

running_jobs_t Running_Jobs;


void Init()
{
   Running_Jobs.jobs_list = (job_t **)NULL;
   Running_Jobs.count = 0;
}

void * Accept_Connections(void * arg)
{
   int sockfd;
   int port = MONITOR_PORT;
   int cli_len;
   struct sockaddr_in serv_addr, cli_addr;

   sockfd = socket(AF_INET, SOCK_STREAM, 0);
   if (sockfd < 0)
   {
      perror("ERROR opening socket");
   }

   bzero((char *) &serv_addr, sizeof(serv_addr));
   serv_addr.sin_family = AF_INET;
   serv_addr.sin_addr.s_addr = INADDR_ANY;
   serv_addr.sin_port = htons(port);

   while (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
   {
      if (errno == EADDRINUSE)
      {
         port ++;
         serv_addr.sin_port = htons(port);
      }
      else
      {
         perror("ERROR on binding");
      }
   }
   fprintf(stderr, "Trace monitor listening for connections on port %d...\n\n", port);

   listen(sockfd, 5);
   cli_len = sizeof(cli_addr);

   while (1)
   {
      int new_sockfd, idx;
      job_t * new_job;

      new_sockfd = accept(sockfd, (struct sockaddr *) &cli_addr, (socklen_t *) &cli_len);
      if (new_sockfd < 0)
      {
         perror("ERROR on accept");
      }

      new_job = (job_t *) malloc(sizeof(job_t));
      new_job -> sockfd = new_sockfd;

      idx = Running_Jobs.count;
      Running_Jobs.jobs_list = (job_t **)realloc(Running_Jobs.jobs_list, (idx + 1) * sizeof(job_t *));
      Running_Jobs.jobs_list[idx] = new_job;
      Running_Jobs.count ++;

      fprintf(stderr, ">>> New job connected to monitor (JobID: %d)\n", new_sockfd);
      fprintf(stderr, "Type how many minutes do you want to trace and press ENTER to start: ");
   }
   return NULL;
}

int main(int argc, char ** argv)
{
   char buf[256];
   int ibuf;

   pthread_t accept_connections;

   Init();

   if (pthread_create(&accept_connections, NULL, Accept_Connections, NULL) != 0)
   { 
      perror("pthread_create");
   }

   do 
   {
      int current_job;
      int job_fd;

      fgets(buf, sizeof(buf), stdin);
      fprintf(stderr, "Read: %s (length %d)\n", buf, (int)strlen(buf));

      current_job = Running_Jobs.count - 1;      
      job_fd = Running_Jobs.jobs_list[current_job]->sockfd;
     
      ibuf = atoi(buf);
      write(job_fd, &ibuf, sizeof(ibuf));
   } while (1);

#if 0
   void * sts;
   if (pthread_join(accept_connections, &sts) != 0)
   {
      perror("pthread_join");
   }
#endif

   return 0;
}

