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
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# ifdef HAVE_FOPEN64
#  define __USE_LARGEFILE64
# endif
# include <stdio.h>
#endif

#include "trace_mode.h"
#include "file_set.h"
#include "record.h"
#include "events.h"
#include "trace_to_prv.h"
#include "object_tree.h"
#include "paraver_state.h"
#include "paraver_generator.h"

int Joint_States = TRUE;
static int Last_State = FALSE;
static int *excluded_states = NULL;
static int num_excluded_states = 0;

int Get_Joint_States (void)
{
	return Joint_States;
}

int Get_Last_State (void)
{
	return Last_State;
}

unsigned int Push_State (unsigned int new_state, unsigned int ptask, unsigned int task, unsigned int thread)
{	
	unsigned int top_state;
	struct thread_t * thread_info;

#if defined(DEBUG_STATES)
	fprintf(stderr, "mpi2prv: DEBUG [T:%d] PUSH_STATE %d\n", task, new_state);
#endif

	/* First event removes the STATE_NOT_TRACING */
	top_state = Top_State(ptask, task, thread);
	if (top_state == STATE_NOT_TRACING)
	{
		Pop_State(STATE_NOT_TRACING, ptask, task, thread);
	}

	thread_info = GET_THREAD_INFO(ptask, task, thread);
	if (thread_info->nStates + 1 >= MAX_STATES)
	{
		fprintf(stderr, "mpi2rpv: Error! MAX states stack reached (%d)\n", thread-1);
		exit(-1);
	}
	thread_info->State_Stack[thread_info->nStates++] = new_state;

	return new_state;
}

unsigned int Pop_State (unsigned int old_state, unsigned int ptask, unsigned int task, unsigned int thread)
{
   unsigned int top_state;
   struct thread_t * thread_info;

#if defined(DEBUG_STATES)
   fprintf(stderr, "mpi2prv: DEBUG [T:%d] POP_STATE\n", task);
#endif

   top_state = Top_State(ptask, task, thread);
   /* Check the top of the stack has the state we are trying to pop (unless STATE_ANY is specified) */
   if ((old_state != STATE_ANY) && (top_state != old_state))
   {
      /* Don't pop if the state isn't pushed! */
      return top_state;
   }

   thread_info = GET_THREAD_INFO(ptask, task, thread);
   if (thread_info->nStates - 1 >= 0)
   {
      top_state = thread_info->State_Stack[--thread_info->nStates];
   }
   else
   {
      top_state = STATE_IDLE;
   }
   
   return top_state;
}

unsigned int Switch_State (unsigned int state, int condition, unsigned int ptask, unsigned int task, unsigned int thread)
{
   if (condition)
   {
      return Push_State (state, ptask, task, thread);
   }
   else
   {
      return Pop_State (state, ptask, task, thread);
   }
}

unsigned int Top_State (unsigned int ptask, unsigned int task, unsigned int thread)
{
   struct thread_t * thread_info;
  
   thread_info = GET_THREAD_INFO(ptask, task, thread);
   if (thread_info->nStates > 0)
   {
      return thread_info->State_Stack[thread_info->nStates - 1];
   }
   else 
   {
      return STATE_IDLE;
   }
}

int State_Excluded (unsigned int state)
{
   int i, excluded = FALSE;

   for (i=0; i<num_excluded_states; i++)
   {
      if (excluded_states[i] == state)
      {
         excluded = TRUE;
         break;
      }
   }
#if defined(DEBUG_STATES)
   fprintf(stderr, "mpi2prv: DEBUG State %d excluded? %d\n", state, excluded);
#endif
   return excluded;
}

void Initialize_Trace_Mode_States (unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread, int mode)
{
   struct thread_t * thread_info; 

#if defined(DEBUG_STATES)
   fprintf(stderr, "[T:%d] WIPE STATES STACK\n", task);
#endif

   /* Clear the states stack */
   thread_info = GET_THREAD_INFO (ptask, task, thread);
   thread_info->nStates = 0;

   /* Push STATE_STOPPED to find it on top of the stack at the APPL_EV EVT_END */
   /* Push_State (STATE_STOPPED, ptask, task, thread); Firstly set at Initialize_States */

   if (mode == TRACE_MODE_BURSTS)
   {
      /* We want the state to appear as IDLE Outside CPU Bursts */
      Push_State (STATE_IDLE, ptask, task, thread);
   }
   else 
   {
      if (thread > 1) 
      {
         /* OpenMP threads will push STATE_RUNNING when executing parallel functions */
         Push_State (STATE_IDLE, ptask, task, thread);
      }
      else 
      {
         /* Tasks appear as RUNNING when not executing parallel code */
         Push_State (STATE_RUNNING, ptask, task, thread);
      }
   }
}

#if defined(DEAD_CODE)
void OLD_Initialize_States (FileSet_t * fset)
{
   int obj;
   unsigned int ptask, task, thread, cpu;
   struct thread_t * thread_info;

   num_excluded_states = 1; 
   excluded_states = (int *)malloc(sizeof(int) * num_excluded_states);
   excluded_states[0] = STATE_IDLE;

   for (obj = 0; obj < num_Files_FS (fset); obj++)
   {
      GetNextObj_FS (fset, obj, &cpu, &ptask, &task, &thread);

      /* Mark no state has been written yet */
      thread_info = GET_THREAD_INFO (ptask, task, thread);
      thread_info->incomplete_state_offset = (off_t)-1;

      if (tracingCircularBuffer())
      {
#if 0
         /* First state is set to STATE_NOT_TRACING */
         Push_State (STATE_NOT_TRACING, ptask, task, thread);
/*
   When using circular buffers, the merger skips all events until the first collective operation.
   This makes the TRACING_MODE_EV to be ommitted, so "Initialize_Trace_Mode_States" (which pops 
   STATE_NOT_TRACING and pushes STATE_RUNNING) is never called. In this way, the application base 
   state is STATE_NOT_TRACING instead of STATE_RUNNING. We could try to solve this forcing to process all 
   events up to MPI_Init END.
   Make note if we process the first events of the trace, then the trace does not start with STATE_NOT_TRACING. 
*/
#else
         Push_State (STATE_RUNNING, ptask, task, thread);
#endif
      }
      else
      {
         /* First state is set to STATE_STOPPED */
         Push_State (STATE_STOPPED, ptask, task, thread);
      }

      /* Write the first state in the trace */
      trace_paraver_state (cpu, ptask, task, thread, 0);
   }
}
#endif

void Initialize_States (FileSet_t * fset)
{
   int obj;
   unsigned int ptask, task, thread, cpu;
   struct thread_t * thread_info;

   num_excluded_states = 1;
   excluded_states = (int *)malloc(sizeof(int) * num_excluded_states);
   excluded_states[0] = STATE_IDLE;

   for (obj = 0; obj < num_Files_FS (fset); obj++)
   {
      GetNextObj_FS (fset, obj, &cpu, &ptask, &task, &thread);

      /* Mark no state has been written yet */
      thread_info = GET_THREAD_INFO (ptask, task, thread);
      thread_info->incomplete_state_offset = (off_t)-1;

      /* First state is set to STATE_STOPPED */
      Push_State (STATE_STOPPED, ptask, task, thread);

      if ((tracingCircularBuffer()) && (getBehaviourForCircularBuffer() == CIRCULAR_SKIP_EVENTS))
      {
		 /* States stack is initialized when TRACING_MODE_EV is processed. 
          * In this tracing behavior The TRACING_MODE_EV is skipped, so we set the initial states here 
          */
         Push_State (STATE_RUNNING, ptask, task, thread);
         Push_State (STATE_NOT_TRACING, ptask, task, thread);
      }

      /* Write the first state in the trace */
      trace_paraver_state (cpu, ptask, task, thread, 0);
   }
}

void Finalize_States (FileSet_t * fset, unsigned long long current_time)
{
   int obj;
   unsigned int ptask, task, thread, cpu;
   
   Last_State = TRUE;

   for (obj = 0; obj < num_Files_FS (fset); obj++)
   {
      GetNextObj_FS (fset, obj, &cpu, &ptask, &task, &thread);

      /* Complete the state record started by the APPL_EV end */
      trace_paraver_state (cpu, ptask, task, thread, current_time);
   }
}
