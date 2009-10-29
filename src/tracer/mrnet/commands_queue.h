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
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __COMMANDS_QUEUE_H__
#define __COMMANDS_QUEUE_H__

/* DEAD_CODE
 * The reason to add the extra "args" field was to allow specyfing command with additional arguments, such as, 
 * START_TRACING and a number of minutes as an additional parameter. But the same can be achieved if who 
 * requests the command, sleeps or waits the same number of minutes and then requests to STOP_TRACING.
 * Another solution is just to enqueue the command, and when it is going to be served, the handler retrieves the 
 * information from the source. I.e.: The monitor sends STOP_TRACING_FOR_THESE_TASKS, and when the FE executes 
 * the handler, it connects to the monitor to retrieve which tasks are going to be disabled. This stalls the 
 * monitor until the command is going to be served, but that even helps for a better synchronization. 
 * This field would make only sense in the front-end side, so the Execute_Command_Handler function protoype
 * in the back-end would not match. In this case, we should change the Execute_Command_Handler function to 
 * return the pointer to the handler to execute, and do the call in the Root/BE functions, having different prototypes.
 */ 

typedef struct
{
	int id;
	unsigned long long time;
#if defined(DEAD_CODE)
	void *args;
#endif
} cmd_t;

void CmdQueue_Initialize ();
#if defined(DEAD_CODE)
void CmdQueue_Insert (int id, void *args);
#else
void CmdQueue_Insert (int id);
#endif
int CmdQueue_FetchArg (void *buf, int size);
int CmdQueue_FetchCmd (cmd_t *command);
#if defined(DEAD_CODE)
void CmdQueue_Free (cmd_t command);
#endif

#endif /* __COMMANDS_QUEUE_H__ */
