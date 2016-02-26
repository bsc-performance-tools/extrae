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

#ifndef __SIGNALS_H__
#define __SIGNALS_H__

#include "config.h"

void SigHandler_FlushAndTerminate (int signum);
void Signals_SetupFlushAndTerminate (int signum);

#if defined(HAVE_ONLINE)
typedef struct
{
    int WaitingForCondition;
    pthread_cond_t WaitCondition;
    pthread_mutex_t ConditionMutex;
} Condition_t;

void Signals_SetupPauseAndResume (int signum_pause, int signum_resume);
void Signals_WaitForPause ();

#if defined(__cplusplus)
extern "C" {
#endif
void Signals_PauseApplication ();
void Signals_ResumeApplication ();

void Signals_CondInit (Condition_t *cond);
void Signals_CondWait (Condition_t *cond);
void Signals_CondWakeUp (Condition_t *cond);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* HAVE_ONLINE */

void Signals_Inhibit ();
void Signals_Desinhibit ();
int Signals_Inhibited ();
void Signals_ExecuteDeferred ();

#endif /* __SIGNALS_H__ */

