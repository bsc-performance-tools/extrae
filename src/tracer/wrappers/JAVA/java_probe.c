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

#include "threadid.h"
#include "wrapper.h"
#include "trace_macros.h"

#include "java_probe.h"

#if 0
# define DEBUG fprintf (stdout, "THREAD %d: %s\n", THREADID, __FUNCTION__);
#else
# define DEBUG
#endif

void Extrae_Java_Thread_start(void)
{
    if(! EXTRAE_ON()) return;

    if (EXTRAE_INITIALIZED() && !Extrae_get_pthread_tracing())
    {
        Backend_Enter_Instrumentation ();

        DEBUG
        TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, JAVA_JVMTI_THREAD_EV, EVT_BEGIN, EMPTY);
        Extrae_AnnotateCPU (LAST_READ_TIME);

        Backend_Leave_Instrumentation ();
    }
}

void Extrae_Java_Thread_end(void)
{
    if(! EXTRAE_ON()) return;

    if (EXTRAE_INITIALIZED() && !Extrae_get_pthread_tracing())
    {
        DEBUG
        TRACE_MISCEVENTANDCOUNTERS(TIME, JAVA_JVMTI_THREAD_EV, EVT_END, EMPTY)
        Extrae_AnnotateCPU (LAST_READ_TIME);

        Backend_Leave_Instrumentation ();
    }
}

void Extrae_Java_Wait_start(void)
{
    if(! EXTRAE_ON()) return;

    if (EXTRAE_INITIALIZED())
    {
        Backend_Enter_Instrumentation ();

        DEBUG
        if (!Extrae_get_pthread_tracing())
            TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, JAVA_JVMTI_WAIT_EV, EVT_BEGIN, EMPTY)
        else
            TRACE_PTHEVENTANDCOUNTERS(LAST_READ_TIME, PTHREAD_JOIN_EV, EVT_BEGIN, EMPTY)
    }
}

void Extrae_Java_Wait_end(void)
{
    if(! EXTRAE_ON()) return;

    if (EXTRAE_INITIALIZED())
    {
        DEBUG
        if (!Extrae_get_pthread_tracing())
            TRACE_MISCEVENTANDCOUNTERS(TIME, JAVA_JVMTI_WAIT_EV, EVT_END, EMPTY)
        else
            TRACE_PTHEVENTANDCOUNTERS(TIME, PTHREAD_JOIN_EV, EVT_END, EMPTY)

        Backend_Leave_Instrumentation ();
    }
}

void Extrae_Java_GarbageCollector_begin(void)
{
	if (EXTRAE_ON())
	{
		Backend_Enter_Instrumentation ();
                TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME,
		  JAVA_JVMTI_GARBAGECOLLECTOR_EV,
		  EVT_BEGIN,
		  EMPTY);
	}
}

void Extrae_Java_GarbageCollector_end(void)
{
	if (EXTRAE_ON())
	{
                TRACE_MISCEVENTANDCOUNTERS(TIME,
		  JAVA_JVMTI_GARBAGECOLLECTOR_EV,
		  EVT_END,
		  EMPTY);
		Backend_Leave_Instrumentation();
	}
}

void Extrae_Java_Exception_begin(void)
{
	if (EXTRAE_ON())
	{
		Backend_Enter_Instrumentation ();
                TRACE_PTHEVENTANDCOUNTERS(LAST_READ_TIME,
		  JAVA_JVMTI_EXCEPTION_EV,
		  EVT_BEGIN,
		  EMPTY);
	}
}

void Extrae_Java_Exception_end(void)
{
	if (EXTRAE_ON())
	{
                TRACE_PTHEVENTANDCOUNTERS(TIME,
		  JAVA_JVMTI_EXCEPTION_EV,
		  EVT_END,
		  EMPTY);
		Backend_Leave_Instrumentation();
	}
}

void Extrae_Java_Object_Alloc (unsigned long long size)
{
	if (EXTRAE_ON())
	{
		Backend_Enter_Instrumentation ();
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME,
		  JAVA_JVMTI_OBJECT_ALLOC_EV,
		  size,
		  EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Java_Object_Free (void)
{
	if (EXTRAE_ON())
	{
		Backend_Enter_Instrumentation ();
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME,
		  JAVA_JVMTI_OBJECT_FREE_EV,
		  EVT_BEGIN,
		  EMPTY);
		Backend_Leave_Instrumentation ();
	}
}
