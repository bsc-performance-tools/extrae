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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/signals.h,v $
 | 
 | @last_commit: $Date: 2009/04/21 10:40:40 $
 | @version:     $Revision: 1.5 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */


#ifndef __SIGNALS_H__
#define __SIGNALS_H__

#include "config.h"

void SigHandler_FlushAndTerminate (int signum);
void Signals_SetupFlushAndTerminate (int signum);

#if defined(HAVE_MRNET)
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

#endif /* HAVE_MRNET */

void Signals_Inhibit ();
void Signals_Desinhibit ();
int Signals_Inhibited ();
void Signals_ExecuteDeferred ();

#endif /* __SIGNALS_H__ */

