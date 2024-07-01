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

#pragma once

#include "wrapper.h"
#include "omp_events.h"

#define ENTERING_INSTRUMENTATION()                  \
{                                                   \
    Backend_Enter_Instrumentation();                \
}

#define ENTERING_OUTLINED()                         \
{                                                   \
    Extrae_OpenMP_Counters();                       \
    Backend_setInInstrumentation (THREADID, FALSE); \
}

#define EXITING_OUTLINED()                          \
{                                                   \
    Backend_setInInstrumentation (THREADID, TRUE);  \
    TIME;                                           \
    Extrae_OpenMP_Counters();                       \
}

#define ENTERING_RUNTIME()                          \
{                                                   \
    if (CURRENT_TRACE_MODE(THREADID) != TRACE_MODE_BURST) \
    { \
      Extrae_OpenMP_Counters();                       \
    } \
}

#define EXITING_RUNTIME()                           \
{                                                   \
    TIME;                                           \
    if (CURRENT_TRACE_MODE(THREADID) != TRACE_MODE_BURST) \
    { \
      Extrae_OpenMP_Counters();                       \
    } \
}

#define EXITING_INSTRUMENTATION()                   \
{                                                   \
    Backend_Leave_Instrumentation();                \
}

