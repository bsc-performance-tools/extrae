/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                   Extrae                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *     ___     This library is free software; you can redistribute it and/or *
 *    /  __         modify it under the terms of the GNU LGPL as published   *
 *   /  /  _____    by the Free Software Foundation; either version 2.1      *
 *  /  /  /     \   of the License, or (at your option) any later version.   *
 * (  (  ( B S C )                                                           *
 *  \  \  \_____/   This library is distributed in hope that it will be      *
 *   \  \__         useful but WITHOUT ANY WARRANTY; without even the        *
 *    \___          implied warranty of MERCHANTABILITY or FITNESS FOR A     *
 *                  PARTICULAR PURPOSE. See the GNU LGPL for more details.   *
 *                                                                           *
 * You should have received a copy of the GNU Lesser General Public License  *
 * along with this library; if not, write to the Free Software Foundation,   *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA          *
 * The GNU LEsser General Public License is contained in the file COPYING.   *
 *                                 ---------                                 *
 *   Barcelona Supercomputing Center - Centro Nacional de Supercomputacion   *
\*****************************************************************************/

#include "common.h"

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

// #define DEBUG_STATES

static int Last_State = FALSE;
static int *excluded_states = NULL;
static int num_excluded_states = 0;

int Get_Last_State (void)
{
	return Last_State;
}

unsigned Push_State (unsigned new_state, unsigned ptask, unsigned task, unsigned thread)
{	
	unsigned int top_state;
	thread_t * thread_info = GET_THREAD_INFO(ptask, task, thread);

#if defined(DEBUG_STATES)
	fprintf(stderr, "mpi2prv: DEBUG [%d:%d:%d] PUSH_STATE %d\n", ptask, task, thread, new_state);
#endif

	/* First event removes the STATE_NOT_TRACING */
	top_state = Top_State(ptask, task, thread);

	if (top_state == STATE_NOT_TRACING)
	{
		if (thread_info->nStates - 1 >= 0)
		{
			thread_info->nStates--;
			top_state = Top_State (ptask, task, thread);
		}
		else
		{
			top_state = STATE_IDLE;
		}
	}

	/* Do we have space to inser the state? If not, allocate it! */
	if (thread_info->nStates == thread_info->nStates_Allocated)
	{
		thread_info->State_Stack = (int*) realloc (thread_info->State_Stack,
		  (thread_info->nStates_Allocated + MAX_STATES_ALLOCATION)*sizeof(int));

		if (thread_info->State_Stack == NULL)
		{
			fprintf (stderr, "mpi2prv: Error! Cannot reallocate state stack for object %d:%d:%d\n", ptask, task, thread);
			exit (-1);
		}
		else
			thread_info->nStates_Allocated += MAX_STATES_ALLOCATION;
	}

	thread_info->State_Stack[thread_info->nStates++] = new_state;

#if defined(DEBUG_STATES)
	fprintf (stderr, "mpi2prv: TOP of the stack is %d, depth is %d\n", new_state, thread_info->nStates);
#endif

	return new_state;
}

unsigned Pop_State (unsigned old_state, unsigned ptask, unsigned task, unsigned thread)
{
   unsigned int top_state;
   thread_t * thread_info = GET_THREAD_INFO(ptask, task, thread);

#if defined(DEBUG_STATES)
   fprintf(stderr, "mpi2prv: DEBUG [%d:%d:%d] POP_STATE\n", ptask, task, thread);
#endif

   /* First event removes the STATE_NOT_TRACING */
   top_state = Top_State(ptask, task, thread);

   if (top_state == STATE_NOT_TRACING)
   {
     if (thread_info->nStates - 1 >= 0)
     {
       thread_info->nStates--;
       top_state = Top_State (ptask, task, thread);
     }
     else
     {
       top_state = STATE_IDLE;
     }
   }

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

#if defined(DEBUG_STATES)
	fprintf (stderr, "mpi2prv: TOP of the stack was %d (current is %d), depth is %d\n", top_state, (thread_info->nStates - 1 >= 0)?thread_info->State_Stack[thread_info->nStates-1]:STATE_IDLE, thread_info->nStates);
#endif
   
   return top_state;
}

static unsigned Pop_State_On_Top(unsigned ptask, unsigned task, unsigned thread)
{
   unsigned int top_state;
   thread_t * thread_info;

   thread_info = GET_THREAD_INFO(ptask, task, thread);
   if (thread_info->nStates - 1 >= 0)
   {
      thread_info->nStates --;
      top_state = Top_State(ptask, task, thread);
   }
   else
   {
      top_state = STATE_IDLE;
   }
   return top_state;
}

unsigned Pop_Until (unsigned until_state, unsigned ptask, unsigned task, unsigned thread)
{
   unsigned int top_state;
   thread_t * thread_info;

   thread_info = GET_THREAD_INFO(ptask, task, thread);

   top_state = Top_State (ptask, task, thread);
   while ((top_state != until_state) && (thread_info->nStates > 0))
   {
      top_state = Pop_State_On_Top (ptask, task, thread);
   }
   return top_state;
}


unsigned Switch_State (unsigned state, int condition, unsigned ptask, unsigned task,
	unsigned thread)
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

unsigned Top_State (unsigned ptask, unsigned task, unsigned thread)
{
   thread_t * thread_info;
  
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

int State_Excluded (unsigned state)
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

void Initialize_Trace_Mode_States (unsigned cpu, unsigned ptask, unsigned task,
	unsigned thread, int mode)
{
	thread_t * thread_info; 

	UNREFERENCED_PARAMETER(cpu);

#if defined(DEBUG_STATES)
	fprintf(stderr, "[%d:%d:%d] WIPE STATES STACK\n", ptask, task, thread);
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
	else if (mode == TRACE_MODE_PHASE_PROFILE)
	{
		Push_State (STATE_PROFILING, ptask, task, thread);
	}
	else if (mode == TRACE_MODE_DISABLED)
	{
		Push_State (STATE_NOT_TRACING, ptask, task, thread);
	}
	else /* Detail */
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

void Dump_States_Stack (unsigned ptask, unsigned task, unsigned thread)
{
  int i = 0;
  thread_t *thread_info = NULL;

  thread_info = GET_THREAD_INFO(ptask, task, thread);
  fprintf(stderr, "Dumping states stack:\n");
  for (i=0; i<thread_info->nStates; i++)
  {
    fprintf(stderr, "STATE %d: %d\n", i, thread_info->State_Stack[i]);
  }
}

void Initialize_States (FileSet_t * fset)
{
   int obj;
   unsigned int ptask, task, thread, cpu;
   thread_t * thread_info;

   num_excluded_states = 1;
   excluded_states = (int *)malloc(sizeof(int) * num_excluded_states);
	if (excluded_states == NULL)
	{
		fprintf(stderr, "mpi2prv: Fatal error! Cannot allocate memory for excluded_states\n");
		exit (-1);
	}
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
      trace_paraver_state_noahead (cpu, ptask, task, thread, current_time);
   }
}
