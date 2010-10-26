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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#include "calltrace.h"
#include "record.h"
#include "trace_macros.h"
#include "wrapper.h"
#include "common_hwc.h"

/* -- El usuario ha desactivado el traceo de MPI callers? -------- */
int Trace_Caller_Enabled[COUNT_CALLER_TYPES] = { 1, 1 };

/* -- Que MPI callers se tracean? -------------------------------- */
int * Trace_Caller[COUNT_CALLER_TYPES] = { NULL, NULL }; 

/* -- Profundidad maxima que necesitamos de la pila de llamadas -- */
int Caller_Deepness[COUNT_CALLER_TYPES] = { 0, 0 };

/* -- Cuantos MPI callers traceamos? ----------------------------- */
int Caller_Count[COUNT_CALLER_TYPES] = { 0, 0 }; 


#if defined(OS_LINUX) || defined(OS_FREEBSD)
 #if !defined(ARCH_IA64) && !defined(ARCH_IA32_x64)
  /* En arquitecturas IA32, se encuentran en la GLIBC y declaradas en <execinfo.h> */
  #ifdef HAVE_EXECINFO_H
  # include <execinfo.h>
  #endif
 #endif
#endif

/* LINUX IA32/PPC o BGL*/
#if (defined(OS_LINUX) && !defined(ARCH_IA64) && !defined(ARCH_IA32_x64)) || defined(OS_FREEBSD) || defined(IS_BG_MACHINE)

void trace_callers (iotimer_t time, int offset, int type) {
   void * callstack[MAX_STACK_DEEPNESS];
   int i, size;
#ifdef MPICALLER_DEBUG
   char **strings; 
#endif

	/* Check for valid CALLER types */
	if (type != CALLER_MPI && type != CALLER_SAMPLING)
		return;

	/* Leave if they aren't initialized (asked by user!) */
	if (Trace_Caller[type] == NULL)
		return;

#if defined(OS_FREEBSD)
	callstack[0] = (void*) trace_callers;
	size = backtrace (&callstack[1], Caller_Deepness[type]+offset-1);
	size++;
#else
	size = backtrace (callstack, Caller_Deepness[type]+offset);
#endif

#ifdef MPICALLER_DEBUG
  /*
   * Para printar aqui el nombre de la funcion, compilar la aplicacion con -rdynamic
   */
   strings = backtrace_symbols (callstack, size);

   printf ("%zd llamadas en la pila.\n", size);
   for (i = 0; i < size; i++) {
      printf ("%s\n", strings[i]);
   }
#endif

	for (i=0; (i<Caller_Deepness[type] && (i+offset-1)<size); i++)
	{
		if (type == CALLER_MPI)
		{
			if (Trace_Caller[CALLER_MPI][i])
			{
				TRACE_EVENT(time, MPI_CALLER_EVENT_TYPE(i+1), (UINT64) callstack[i+offset-1]);
			}
		}
#if defined(SAMPLING_SUPPORT)
		else if (type == CALLER_SAMPLING)
		{
			if (Trace_Caller[CALLER_SAMPLING][i])
			{
				SAMPLE_EVENT_NOHWC(time, SAMPLING_EV+i+1, (UINT64) callstack[i+offset-1]);
			}
		}
#endif
	}	  
}

UINT64 get_caller (int offset)
{
	void * callstack[MAX_STACK_DEEPNESS];
	int size;
#if defined(DEBUG)
	int i;
#endif

#if defined(OS_FREEBSD)
	callstack[0] = (void*) get_caller;
	size = backtrace (&callstack[1], offset-1);
	size++;
#else
	size = backtrace (callstack, offset);
#endif

#if defined(DEBUG)
	for (i = 0; i < size; i++)
		fprintf (stderr, "DEBUG: depth %d address %p %c\n", i, callstack[i],(offset-1 == i)?'*':' ');
#endif

	return (UINT64) ((offset-1 >= size)?0:callstack[offset-1]);
}

#endif /* LINUX IA32 */

/* LINUX IA64 */
#if defined(OS_LINUX) && (defined(ARCH_IA64) || defined(ARCH_IA32_x64))

#if defined(UNWIND_SUPPORT)
# define UNW_LOCAL_ONLY
# ifdef HAVE_LIBUNWIND_H
#  include <libunwind.h>
# endif
#endif

void trace_callers (iotimer_t time, int offset, int type) {
#if defined(UNWIND_SUPPORT)
	int current_deep = 1;
	unw_cursor_t cursor;
	unw_context_t uc;
	unw_word_t ip;
#if defined(MPICALLER_DEBUG)
	unw_word_t sp;
#endif

	/* Check for valid CALLER types */
	if (type != CALLER_MPI && type != CALLER_SAMPLING)
		return;

	/* Leave if they aren't initialized (asked by user!) */
	if (Trace_Caller[type] == NULL)
		return;
  
	unw_getcontext(&uc);
	unw_init_local(&cursor, &uc);

	offset --; /* Don't compute call to unw_getcontext */
	while ((unw_step(&cursor) > 0) && (current_deep < Caller_Deepness[type]+offset))
	{
		unw_get_reg(&cursor, UNW_REG_IP, &ip);
#if defined(MPICALLER_DEBUG)
		if (current_deep >= offset)
		{
			unw_get_reg(&cursor, UNW_REG_SP, &sp);
			fprintf (stderr, "(%d) ip = %lx, sp = %lx\n", current_deep, (long) ip, (long) sp);
		}
#endif
    
		if (current_deep >= offset)
		{
			if (type == CALLER_MPI)
			{
				if (Trace_Caller[CALLER_MPI][current_deep-offset])
				{
					TRACE_EVENT(time, MPI_CALLER_EVENT_TYPE(current_deep-offset+1), (UINT64)ip);
				}
			}
#if defined(SAMPLING_SUPPORT)
			else if (type == CALLER_SAMPLING)
			{
				if (Trace_Caller[CALLER_SAMPLING][current_deep-offset])
				{
					SAMPLE_EVENT_NOHWC(time, SAMPLING_EV+current_deep-offset+1, (UINT64) ip);
				}
			} 
#endif
		}
		current_deep ++;
	}
#else /* UNWIND_SUPPORT */
	UNREFERENCED_PARAMETER(time);
	UNREFERENCED_PARAMETER(offset);
	UNREFERENCED_PARAMETER(type);
#endif /* UNWIND_SUPPORT */
}

UINT64 get_caller (int offset)
{
#if defined(UNWIND_SUPPORT)
	int current_deep = 0;
	unw_cursor_t cursor;
	unw_context_t uc;
	unw_word_t ip;

	unw_getcontext(&uc);
	unw_init_local(&cursor, &uc);

	offset --; /* Don't compute call to unw_getcontext */
	while (current_deep < offset)
	{
		unw_get_reg(&cursor, UNW_REG_IP, &ip);
#if defined(DEBUG)
		fprintf (stderr, "DEBUG: depth %d address %08llx %c\n", current_deep, ip, (offset-1 == current_deep)?'*':' ');
#endif
		if (unw_step (&cursor) <= 0)
			return 0;
		current_deep ++;
	}
	return (UINT64) ip;
#else /* UNWIND_SUPPORT */
	UNREFERENCED_PARAMETER(offset);
#endif /* UNWIND_SUPPORT */
	return 0;
}

#endif /* LINUX IA64 */

#if defined(OS_DEC) 

#ifdef HAVE_EXCPT_H
# include <excpt.h>
#endif
#ifdef HAVE_PDSC_H
# include <pdsc.h>
#endif

void trace_mpi_callers(iotimer_t time, int offset) {
   CONTEXT contexto;
   int rc = 0, actual_deep = 1;

	/* Leave if they aren't initialized (asked by user!) */
	if (Trace_Caller[CALLER_MPI] == NULL)
		return;
  
   exc_capture_context (&contexto);
   /* Unwind de la pila*/
   while (!rc && contexto.sc_pc && actual_deep <= Caller_Deepness[type]+offset)  {
	  
#ifdef MPICALLER_DEBUG                                                                           
       /* Tampoco tenemos backtrace_symbols. Solo printamos las direcciones. */                   
       fprintf(stderr, "[%d] 0x%012lx\n", actual_deep, contexto.sc_pc);                              
#endif        

       if (actual_deep >= offset)
          if (Trace_MPI_Caller[actual_deep-offset])
          {
            TRACE_EVENT(time, MPI_CALLER_EVENT_TYPE(actual_deep-offset+1), (UINT64)contexto.sc_pc);
          }
      
       rc = exc_virtual_unwind(0, &contexto);
       actual_deep ++;
   }	    
}

#endif /* DEC */

#if defined(OS_AIX)

#ifdef HAVE_UCONTEXT_H
# include <ucontext.h>
#endif

void trace_callers(iotimer_t time, int offset, int type) {
	ucontext_t Contexto;
	void *  InstructionPointer;
	void ** StackFrame;
	int Frame = 1;

	/* Check for valid CALLER types */
	if (type != CALLER_MPI && type != CALLER_SAMPLING)
		return;

	/* Leave if they aren't initialized (asked by user!) */
	if (Trace_Caller[type] == NULL)
		return;

	if (getcontext (&Contexto) < 0) {
		fprintf(stderr, "backtrace: Error retrieving process context.\n");
		return;
	}
#if defined(MPI_CALLER_DEBUG)
	fprintf(stderr, "Executing instruction at 0x%08x\n", (&Contexto)->uc_mcontext.jmp_context.iar);
#endif

	/* Roll down the stack frames */
	InstructionPointer = (void *) (&Contexto)->uc_mcontext.jmp_context.iar;
	StackFrame = (void **) (&Contexto)->uc_mcontext.jmp_context.gpr[1];

	while ((StackFrame) && (Frame < Caller_Deepness[type]+offset)) {
#if defined(MPI_CALLER_DEBUG)
		fprintf(stderr, "(%2d) 0x%p\n", Frame, InstructionPointer);
#endif
		if (Frame >= offset) {
			if (Trace_Caller[type][Frame-offset]) {
				/* Esta es la profundidad que queremos tracear */
				TRACE_EVENT(time, MPI_CALLER_EVENT_TYPE(Frame-offset+1), (UINT64)InstructionPointer);
			} 
		}

		if (!StackFrame[0]) return;

		/* Seguramente esta comprobacion se puede comentar */
#if defined(MPI_CALLER_DEBUG)
		if (!ValidAddress((void *) StackFrame) ||
			!ValidAddress((void *) StackFrame[0]) ||
			!ValidAddress((void *) ((void **) StackFrame[0] + 2)))
		{
			fprintf(stderr, "backtrace: Invalid frame at %p\n", (void *)StackFrame);
			return;
		}
#endif
		StackFrame = (void **) StackFrame[0];
		InstructionPointer = StackFrame[2];
		Frame ++;
	}
}

UINT64 get_caller (int offset)
{
	/* TODO */
	/* AIX ucontext */
   return 0;
}

#ifdef MPI_CALLER_DEBUG
static int ValidAddress (void * Addr) {
	return (access((char *) Addr, F_OK) && (errno == EFAULT))?0:1;
}
#endif

#endif /* OS_AIX */

#if defined (OS_SOLARIS)
void trace_callers (iotimer_t time, int offset, int type)
{
	/* TODO */
	/* Solaris walkcontext */
}

UINT64 get_caller (int offset)
{
	return 0;
}
#endif

