/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/mrnet/commands_queue.C,v $
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#ifdef HAVE_PTHREAD_H
# include <pthread.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_SYS_TIME_H
# include <sys/time.h>
#endif
#include "commands_queue.h"
#include "common.h"
#include "utils.h"

int CommandsQueue[2];
int CmdQueue_Initialized  = FALSE;
pthread_mutex_t QueueLock = PTHREAD_MUTEX_INITIALIZER;

/**
 * Initializes the commands queue 
 */
void CmdQueue_Initialize ()
{
	pipe (CommandsQueue);
	CmdQueue_Initialized = TRUE;
}

/**
 * Inserts atomically a command in the queue with the current timestamp and extra arguments.
 * @param[in] id The command identifier.
 * @param[in] args The extra arguments the command may need.
 */ 
#if defined(DEAD_CODE)
void CmdQueue_Insert (int id, void *args)
#else
void CmdQueue_Insert (int id)
#endif
{
	if (CmdQueue_Initialized)
	{
		cmd_t command;
		struct timeval t;

		pthread_mutex_lock (&QueueLock);

		command.id = id;
		gettimeofday(&t, 0);
		command.time = (t.tv_sec * 1000000) + t.tv_usec;
#if defined(DEAD_CODE)
		command.args = args;
#endif

		write(CommandsQueue[1], &command, sizeof(cmd_t));

		pthread_mutex_unlock (&QueueLock);
	}
}

int CmdQueue_FetchArg (void *buf, int size)
{
	if (CmdQueue_Initialized)
	{
		read (CommandsQueue[0], buf, size);
		return 1;
	}
	return 0;
}

/**
 * Fetches the next command of the queue.
 * @param[inout] command The command from the queue is stored in this buffer.
 * @return The command identifier.
 */
int CmdQueue_FetchCmd (cmd_t *command)
{
	return CmdQueue_FetchArg (command, sizeof(cmd_t));
}

#if defined(DEAD_CODE)
/** 
 * Free all memory allocated for the extra arguments of the command
 */
void CmdQueue_Free (cmd_t command)
{
    xfree(command.args);
}
#endif
