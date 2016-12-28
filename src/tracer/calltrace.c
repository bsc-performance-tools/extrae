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

#include "calltrace.h"
#include "record.h"
#include "trace_macros.h"
#include "wrapper.h"
#include "common_hwc.h"

//#define DEBUG
//#define MPICALLER_DEBUG

/* -- El usuario ha desactivado el traceo de MPI callers? -------- */
int Trace_Caller_Enabled[COUNT_CALLER_TYPES] = { TRUE, TRUE, TRUE, TRUE, TRUE };

/* -- Que MPI callers se tracean? -------------------------------- */
int * Trace_Caller[COUNT_CALLER_TYPES] = { NULL, NULL, NULL, NULL, NULL }; 

/* -- Profundidad maxima que necesitamos de la pila de llamadas -- */
int Caller_Deepness[COUNT_CALLER_TYPES] = { 0, 0, 0, 0, 0 };

/* -- Cuantos MPI callers traceamos? ----------------------------- */
int Caller_Count[COUNT_CALLER_TYPES] = { 0, 0, 0, 0, 0 }; 

#if defined(UNWIND_SUPPORT)

# define UNW_LOCAL_ONLY
# ifdef HAVE_LIBUNWIND_H
#  include <libunwind.h>
# endif

void Extrae_trace_callers (iotimer_t time, int offset, int type)
{
	int current_deep = 1;
	unw_cursor_t cursor;
	unw_context_t uc;
	unw_word_t ip;

	/* Leave if they aren't initialized (asked by user!) */
	if (Trace_Caller[type] == NULL)
		return;
  
	if (unw_getcontext(&uc) < 0)
		return;

	if (unw_init_local(&cursor, &uc) < 0)
		return;

	offset --; /* Don't compute call to unw_getcontext */
	while ((unw_step(&cursor) > 0) && (current_deep < Caller_Deepness[type]+offset))
	{
		if (unw_get_reg(&cursor, UNW_REG_IP, &ip) < 0)
			break;

#if defined(MPICALLER_DEBUG)
		if (current_deep >= offset)
			fprintf (stderr, "emitted (deep = %d, offset = %d) ip = %lx\n", current_deep, offset, (long) ip);
		else
			fprintf (stderr, "ignored (deep = %d, offset = %d) ip = %lx\n", current_deep, offset, (long) ip);
#endif
    
		if (current_deep >= offset)
		{
			if (type == CALLER_MPI || type == CALLER_DYNAMIC_MEMORY || type == CALLER_IO || type == CALLER_SYSCALL)
			{
				if (Trace_Caller[type][current_deep-offset])
				{
					TRACE_EVENT(time, CALLER_EVENT_TYPE(type, current_deep-offset+1), (UINT64)ip);
				}
			}
#if defined(SAMPLING_SUPPORT)
			else if (type == CALLER_SAMPLING)
			{
				if (Trace_Caller[type][current_deep-offset])
					SAMPLE_EVENT_NOHWC(time, SAMPLING_EV+current_deep-offset+1, (UINT64) ip);
			} 
#endif
		}
		current_deep ++;
	}
}

UINT64 Extrae_get_caller (int offset)
{
	int current_deep = 0;
	unw_cursor_t cursor;
	unw_context_t uc;
	unw_word_t ip;

	if (unw_getcontext(&uc) < 0)
		return 0;

	if (unw_init_local(&cursor, &uc))
		return 0;

	offset --; /* Don't compute call to unw_getcontext */
	while (current_deep <= offset)
	{
		if (unw_get_reg(&cursor, UNW_REG_IP, &ip) < 0)
			break;

#if defined(DEBUG)
		fprintf (stderr, "DEBUG: offset %d depth %d address %08llx %c\n", offset, current_deep, ip, (offset == current_deep)?'*':' ');
#endif
		if (unw_step (&cursor) <= 0)
			return 0;
		current_deep ++;
	}
	return (UINT64) ip;
}

#else /* UNWIND_SUPPORT */

# if defined(OS_LINUX) || defined(OS_FREEBSD) || defined(OS_DARWIN)
#  if !defined(ARCH_IA64)
#   ifdef HAVE_EXECINFO_H
#    include <execinfo.h>
#   endif
#  endif
# endif

/* LINUX IA32/PPC o BGL*/
# if (defined(OS_LINUX) && !defined(ARCH_IA64)) || defined(OS_FREEBSD) || defined(OS_DARWIN) || defined(IS_BG_MACHINE)

void Extrae_trace_callers (iotimer_t time, int offset, int type) {
	void * callstack[MAX_STACK_DEEPNESS];
	int size;
	int frame;
#ifdef MPICALLER_DEBUG
	int i;
	char **strings; 
#endif

	/* Leave if they aren't initialized (asked by user!) */
	if (Trace_Caller[type] == NULL)
		return;

#if (defined(OS_DARWIN) || defined(OS_FREEBSD)) && defined (HAVE_EXECINFO_H)
	callstack[0] = (void*) Extrae_trace_callers;
	size = backtrace (&callstack[1], Caller_Deepness[type]+offset-1);
	size++;
#else
	size = backtrace (callstack, Caller_Deepness[type]+offset);
#endif

#ifdef MPICALLER_DEBUG
	/* To print the name of the function, compile with -rdynamic */
	strings = backtrace_symbols (callstack, size);

	fprintf (stderr, "%d calls in the callstack.\n", size);
	for (i = 0; i < size; i++)
		printf ("%s\n", strings[i]);
#endif

	for (frame = 0; ((frame < Caller_Deepness[type]+offset-1) && (frame < size)); frame ++)
	{
	  int current_caller = frame - offset + 2;          

#ifdef MPICALLER_DEBUG
		fprintf(stderr, "#%d ip=%lx", frame, (long)callstack[frame]);
		if (current_caller > 0)
			fprintf(stderr, " current_caller=%d trace_this_caller?=%d", current_caller, Trace_Caller[type][current_caller - 1]);
		fprintf(stderr, "\n");
#endif
		if (current_caller > 0)
		{
			if (type == CALLER_MPI || type == CALLER_DYNAMIC_MEMORY || type == CALLER_IO || type == CALLER_SYSCALL)
			{
				if (Trace_Caller[type][current_caller - 1])
					TRACE_EVENT(time, CALLER_EVENT_TYPE(type, current_caller),
					  (UINT64) callstack[frame]);
			}
#if defined(SAMPLING_SUPPORT)
			else if (type == CALLER_SAMPLING)
			{
				if (Trace_Caller[CALLER_SAMPLING][current_caller - 1])
					SAMPLE_EVENT_NOHWC(time, SAMPLING_EV+current_caller,
					  (UINT64) callstack[frame]);
			}
#endif
		}
	}
}

UINT64 Extrae_get_caller (int offset)
{
	void * callstack[MAX_STACK_DEEPNESS];
	int size;
#if defined(DEBUG)
	int i;
#endif

#if (defined(OS_DARWIN) || defined(OS_FREEBSD)) && defined(HAVE_EXECINFO_H)
	callstack[0] = (void*) Extrae_get_caller;
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

# endif /* LINUX IA32 */

# if defined(OS_LINUX) && defined(ARCH_IA64)
void Extrae_trace_callers (iotimer_t time, int offset, int type)
{
	UNREFERENCED_PARAMETER(time);
	UNREFERENCED_PARAMETER(offset);
	UNREFERENCED_PARAMETER(type);

	/* IA64 requires unwind! */

	return;
}

UINT64 Extrae_get_caller (int offset)
{
	UNREFERENCED_PARAMETER(offset);

	/* IA64 requires unwind! */

	return 0;
}
# endif /* LINUX IA64 */

#if defined(OS_DEC) 

#ifdef HAVE_EXCPT_H
# include <excpt.h>
#endif
#ifdef HAVE_PDSC_H
# include <pdsc.h>
#endif

#error "This code is unmantained! If you reach this, contact with tools@bsc.es"

void trace_mpi_callers(iotimer_t time, int offset, int type)
{
	CONTEXT ctx;
	int rc = 0, actual_deep = 1;

	/* Leave if they aren't initialized (asked by user!) */
	if (Trace_Caller[CALLER_MPI] == NULL)
		return;
  
	exc_capture_context (&ctx);
	/* Unwind de la pila*/
	while (!rc && ctx.sc_pc && actual_deep <= Caller_Deepness[type]+offset)
	{
	  
#ifdef MPICALLER_DEBUG                                                                           
		/* Tampoco tenemos backtrace_symbols. Solo printamos las direcciones. */                   
		fprintf(stderr, "[%d] 0x%012lx\n", actual_deep, ctx.sc_pc);                              
#endif        

		if (actual_deep >= offset)
			if (Trace_MPI_Caller[actual_deep-offset])
				TRACE_EVENT(time, MPI_CALLER_EVENT_TYPE(actual_deep-offset+1),
				  (UINT64)ctx.sc_pc);
		rc = exc_virtual_unwind(0, &ctx);
		actual_deep ++;
	}	    
}

#endif /* DEC */

#if defined(OS_AIX)

#ifdef HAVE_UCONTEXT_H
# include <ucontext.h>
#endif

#error "This code is unmantained! If you reach this, contact with tools@bsc.es"

void Extrae_trace_callers(iotimer_t time, int offset, int type)
{
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

UINT64 Extrae_get_caller (int offset)
{
	UNREFERENCED_PARAMETER(offset);

	/* TODO */
	/* AIX ucontext */
   return 0;
}

#if defined(MPI_CALLER_DEBUG)
static int ValidAddress (void * Addr) {
	return (access((char *) Addr, F_OK) && (errno == EFAULT))?0:1;
}
#endif

#endif /* OS_AIX */

#if defined (OS_SOLARIS)
void Extrae_trace_callers (iotimer_t time, int offset, int type)
{
	/* TODO */
	/* Solaris walkcontext */

	UNREFERENCED_PARAMETER(time);
	UNREFERENCED_PARAMETER(offset);
	UNREFERENCED_PARAMETER(type);
}

UINT64 Extrae_get_caller (int offset)
{
	UNREFERENCED_PARAMETER(offset);

	return 0;
}
#endif

#endif /* UNWIND_SUPPORT */

