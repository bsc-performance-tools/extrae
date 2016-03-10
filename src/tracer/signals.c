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

#include <config.h>

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_SIGNAL_H
# include <signal.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#include "signals.h"
#include "utils.h"
#include "wrapper.h"
#if defined(HAVE_ONLINE)
# ifdef HAVE_PTHREAD_H
#  include <pthread.h>
# endif
#endif

/* #define DBG_SIGNALS */

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
#if _XOPEN_SOURCE >= 700 || _POSIX_C_SOURCE >= 200809L
		fprintf (stderr, PACKAGE_NAME": Attention! Signal %d (%s) caugth. Flushing buffer to disk and terminating\n",
		  signum, strsignal (signum));
#else
		fprintf (stderr, PACKAGE_NAME": Attention! Signal %d caugth. Flushing buffer to disk and terminating\n",
		  signum);
#endif
		Backend_Finalize ();
		exit (0);
	}
	else
	{
#if _XOPEN_SOURCE >= 700 || _POSIX_C_SOURCE >= 200809L
		fprintf (stderr, PACKAGE_NAME": Attention! Signal %d (%s) caught. Notifying to flush buffers whenever possible.\n",
		  signum, strsignal (signum));
#else
		fprintf (stderr, PACKAGE_NAME": Attention! Signal %d caught. Notifying to flush buffers whenever possible.\n",
		  signum);
#endif
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

#if defined(HAVE_ONLINE)

pthread_t MainApplThread;

int signum_pause, signum_resume;
sigset_t pause_set, resume_set;

/* -----------------------------------------------------------------------
 * SigHandler_PauseApplication
 * ----------------------------------------------------------------------- */

void SigHandler_PauseApplication (int signum)
{
	UNREFERENCED_PARAMETER(signum);

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
	UNREFERENCED_PARAMETER(signum);

#if defined(DBG_SIGNALS)
	fprintf(stderr, "[SigHandler_ResumeApplication] Application RESUMED\n");
#endif
}

/* ----------------------------------------
 * Signals_SetupPauseAndResume
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

#endif /* HAVE_ONLINE */

