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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/signals.c,v $
 | 
 | @last_commit: $Date: 2009/04/21 10:40:40 $
 | @version:     $Revision: 1.7 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: signals.c,v 1.7 2009/04/21 10:40:40 gllort Exp $";

#include <config.h>

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_SIGNAL_H
# include <signal.h>
#endif
#include "signals.h"
#include "utils.h"
#include "wrapper.h"

#define DBG_SIGNALS

/* -----------------------------------------------------------------------
 * SigHandler_FlushAndTerminate
 * Flushes the buffers to disk and disables tracing
 * ----------------------------------------------------------------------- */

int sigInhibited = FALSE;

void Signals_Inhibit()
{
	sigInhibited = TRUE;
}

void Signals_Desinhibit()
{
	sigInhibited = FALSE;
}

int Signals_Inhibited()
{
	return sigInhibited;
}

int Deferred_Signal_FlushAndTerminate = FALSE;

void SigHandler_FlushAndTerminate (int signum)
{
	/* We don't need to reprogram the signal, it must happen only once! */

	if (!Signals_Inhibited())
	{
		/* Flush buffer to disk */
		fprintf (stderr, "SIGNAL %d received: Flushing buffer to disk\n", signum);

		Thread_Finalization();

		/* Disable further tracing */
		fprintf (stderr, "TASK %d has flushed the buffer.\n", TASKID);
		mpitrace_on = 0;
	}
	else
	{
		fprintf (stderr, "SIGNAL %d received... notifying to flush buffers\n", signum);
		Deferred_Signal_FlushAndTerminate = 1;
	}
}

void Signals_ExecuteDeferred ()
{
	if (Deferred_Signal_FlushAndTerminate)
	{
		SigHandler_FlushAndTerminate(0);
	}
}

/* ----------------------------------------
 * Signals_SetupFlushAndTerminate
 * Assign the appropriate signal handlers 
 * ---------------------------------------- */

void Signals_SetupFlushAndTerminate (int signum)
{
    signal (signum, SigHandler_FlushAndTerminate);
}

#if defined(HAVE_MRNET)

#include <pthread.h>

pthread_t MainApplThread;

int signum_pause, signum_resume;
sigset_t pause_set, resume_set;

/* -----------------------------------------------------------------------
 * SigHandler_PauseApplication
 * ----------------------------------------------------------------------- */

void SigHandler_PauseApplication (int signum)
{
#if defined(DBG_SIGNALS)
	fprintf(stderr, "[SigHandler_PauseApplication] Application PAUSED\n");
#endif
	sigsuspend (&resume_set);
}

/* -----------------------------------------------------------------------
 * SigHandler_ResumeApplication
 * ----------------------------------------------------------------------- */

void SigHandler_ResumeApplication (int signum)
{
#if defined(DBG_SIGNALS)
	fprintf(stderr, "[SigHandler_ResumeApplication] Application RESUMED\n");
#endif
}

/* ----------------------------------------
 * Signals_Signals_SetupPauseAndResume
 * Assign the appropriate signal handlers 
 * ---------------------------------------- */

void Signals_SetupPauseAndResume (int signum1, int signum2)
{
	struct sigaction sigact_pause, sigact_resume;

#if defined(DBG_SIGNALS)
    fprintf(stderr, "[Signals_SetupPauseAndResume] Setting up Pause/Resume signals\n");
#endif

	signum_pause  = signum1;
	signum_resume = signum2;

	MainApplThread = pthread_self();

	sigemptyset( &sigact_pause.sa_mask );
	sigact_pause.sa_flags = 0;
	sigact_pause.sa_handler = SigHandler_PauseApplication;
	sigaction (signum_pause, &sigact_pause, NULL);
	sigfillset( &pause_set );
	sigdelset( &pause_set, signum_pause );

    sigemptyset( &sigact_resume.sa_mask );
    sigact_resume.sa_flags = 0;
    sigact_resume.sa_handler = SigHandler_ResumeApplication;
    sigaction (signum_resume, &sigact_resume, NULL);
    sigfillset( &resume_set );
    sigdelset( &resume_set, signum_resume );
}

/* -----------------------------------------------------------------------
 * Signals_PauseApplication
 * Signals_ResumeApplication
 * Signals_WaitForPause
 * Pause/Resume the application
 * ----------------------------------------------------------------------- */

void Signals_PauseApplication ()
{
    pthread_kill(MainApplThread, signum_pause);
}

void Signals_ResumeApplication ()
{
    pthread_kill(MainApplThread, signum_resume);
}

void Signals_WaitForPause ()
{
    sigsuspend (&pause_set);
}

#if 0
int             WaitingForCondition;
pthread_cond_t  WaitCondition;
pthread_mutex_t ConditionMutex;

void Signals_CondWait ()
{
	int rc; 
	rc = pthread_mutex_lock(&ConditionMutex);
	WaitingForCondition = TRUE;
	while (WaitingForCondition)
	{
		rc = pthread_cond_wait(&WaitCondition, &ConditionMutex);
	}
	rc = pthread_mutex_unlock(&ConditionMutex);
}

void Signals_CondWakeUp ()
{
	int rc; 
	rc = pthread_mutex_lock(&ConditionMutex);
	WaitingForCondition = FALSE;
	rc = pthread_cond_signal(&WaitCondition);
	rc = pthread_mutex_unlock(&ConditionMutex);
}
#endif

void Signals_CondInit (Condition_t *cond)
{
  	pthread_mutex_init(&(cond->ConditionMutex), NULL);
	pthread_cond_init(&(cond->WaitCondition), NULL);
	cond->WaitingForCondition = TRUE;
}

void Signals_CondWait (Condition_t *cond)
{
	pthread_mutex_lock(&(cond->ConditionMutex));
	while (cond->WaitingForCondition)
	{
		pthread_cond_wait(&(cond->WaitCondition), &(cond->ConditionMutex));
	}
	pthread_mutex_unlock(&(cond->ConditionMutex));
}

void Signals_CondWakeUp (Condition_t *cond)
{
	pthread_mutex_lock(&(cond->ConditionMutex));
	cond->WaitingForCondition = FALSE;
	pthread_cond_signal(&(cond->WaitCondition));
	pthread_mutex_unlock(&(cond->ConditionMutex));
}

#endif /* HAVE_MRNET */

