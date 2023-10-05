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

#ifdef HAVE_SYS_TIME_H
# include <sys/time.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_UCONTEXT_H
# include <ucontext.h>
#endif

#ifdef OS_ANDROID
#warning "In case NDK doesn't provide ucontext you can download it from https://google-breakpad.googlecode.com/svn-history/r1000/trunk/src/client/linux/android_ucontext.h"
# include <asm/android_ucontext.h>
#endif

#ifdef HAVE_SIGNAL_H
# include <signal.h>
#endif

#include "sampling-common.h"
#include "sampling-timer.h"
#include "trace_macros.h"
#include "threadid.h"
#include "wrapper.h"
#include "utils.h"
#include "xalloc.h"

#if !defined(OS_RTEMS)

#if defined(SAMPLING_SUPPORT)
int SamplingSupport = FALSE;
static int SamplingRunning = FALSE;
#endif

static struct sigaction signalaction;

void Extrae_SamplingHandler (void* address)
{
	if (tracejant && Extrae_isSamplingEnabled() && !Backend_inInstrumentation(THREADID))
	{
		Backend_setInSampling(THREADID, TRUE);
		UINT64 temps = Clock_getCurrentTime_nstore();
		SAMPLE_EVENT_HWC (temps, SAMPLING_EV, (unsigned long long) address);
		Extrae_trace_callers (temps, 6, CALLER_SAMPLING);
		Backend_setInSampling(THREADID, FALSE);
	}
}

void Extrae_SamplingHandler_PAPI (void* address)
{
	if (tracejant && Extrae_isSamplingEnabled() && !Backend_inInstrumentation(THREADID))
	{
		Backend_setInSampling(THREADID, TRUE);
		UINT64 temps = Clock_getCurrentTime_nstore();
		SAMPLE_EVENT_HWC (temps, SAMPLING_EV, (unsigned long long) address);
		Extrae_trace_callers (temps, 8, CALLER_SAMPLING);
		Backend_setInSampling(THREADID, FALSE);
	}
}

#if defined(IS_BGP_MACHINE) || defined(IS_BGQ_MACHINE)
/* BG/P  & BG/Q */
# if __WORDSIZE == 32
#  define UCONTEXT_REG(uc, reg) ((uc)->uc_mcontext.uc_regs->gregs[reg])
# else
#  define UCONTEXT_REG(uc, reg) ((uc)->uc_mcontext.gp_regs[reg])
# endif
# define PPC_REG_PC 32
#elif defined(IS_BGL_MACHINE)
# error "Don't know how to access the PC! Check if it is like BG/P"
#endif

static unsigned long long Sampling_variability;
static struct itimerval SamplingPeriod_base;
static struct itimerval SamplingPeriod;
static int SamplingClockType;

static void PrepareNextAlarm (void)
{
	/* Set next timer! */
	if (Sampling_variability > 0)
	{
		unsigned long long v = xtr_random() % (Sampling_variability);
		unsigned long long s, us;

		us = (v + SamplingPeriod_base.it_value.tv_usec) % 1000000;
		s = (v + SamplingPeriod_base.it_value.tv_usec) / 1000000 + SamplingPeriod_base.it_interval.tv_sec;

		SamplingPeriod.it_interval.tv_sec = 0;
		SamplingPeriod.it_interval.tv_usec = 0;
		SamplingPeriod.it_value.tv_usec = us;
		SamplingPeriod.it_value.tv_sec = s;
	}
	else
		SamplingPeriod = SamplingPeriod_base;

	setitimer (SamplingClockType ,&SamplingPeriod, NULL);
}

static void TimeSamplingHandler (int sig, siginfo_t *siginfo, void *context)
{
	caddr_t pc;

	/* 
	 * This macro is subsituted by the definition of the struct ucontext as
	 * found by the configure macro AX_CHECK_UCONTEXT. We found that in
	 * newer versions of glibc the type changes from struct ucontext to
	 * ucontext_t
	 */
	STRUCT_UCONTEXT *uc = (STRUCT_UCONTEXT *) context;

// Last check required until we can use sigcontext sc_regs struct
#if (defined(OS_LINUX) || defined(OS_ANDROID)) && !defined(ARCH_RISCV64)
	struct sigcontext *sc = (struct sigcontext *) &uc->uc_mcontext;
#endif

	UNREFERENCED_PARAMETER(sig);
	UNREFERENCED_PARAMETER(siginfo);

#if defined(IS_BGP_MACHINE) || defined(IS_BGQ_MACHINE)
	pc = (caddr_t)UCONTEXT_REG(uc, PPC_REG_PC);
#elif defined(OS_LINUX) || defined(OS_ANDROID)
# if defined(ARCH_IA32) && !defined(ARCH_IA32_x64)
	pc = (caddr_t)sc->eip;
# elif defined(ARCH_IA32) && defined(ARCH_IA32_x64)
	pc = (caddr_t)sc->rip;
# elif defined(ARCH_IA64)
	pc = (caddr_t)sc->sc_ip;
# elif defined(ARCH_PPC)
	pc = (caddr_t)sc->regs->nip;
# elif defined(ARCH_ARM) && !defined(ARCH_ARM64)
	pc = (caddr_t)sc->arm_pc;
# elif defined(ARCH_ARM) && defined(ARCH_ARM64)
	pc = (caddr_t)sc->pc;
# elif defined(ARCH_SPARC64)
    //pc = (caddr_t)sc->sigc_regs->tpc;
    //pc = (caddr_t)sc->mc_gregs->tpc;
    //pc = (caddr_t)sc->mc_gregs[MC_PC];
	pc = 0;
# elif defined(ARCH_RISCV64)
	/*
	 * Directly read the PC register using ucontext
	 * [1] https://reviews.llvm.org/D66870
	 * [2] https://sourceware.org/git/?p=glibc.git;a=blob;f=sysdeps/unix/sysv/linux/riscv/sys/ucontext.h
	 */
	pc = (caddr_t)uc->uc_mcontext.__gregs[REG_PC];
	/*
	 * In the future it may be possible to obtain the PC through sigcontext
	 * sc_regs struct. Right now it is empty, check [3].
	 * [1] https://linuxplumbersconf.org/event/7/contributions/811/attachments/613/1103/An_introduction_of_vector_support_in_RISC-V_Linux.pdf
	 * [2] https://github.com/torvalds/linux/blob/master/arch/riscv/include/uapi/asm/sigcontext.h
	 * [3] https://sourceware.org/git/?p=glibc.git;a=blob;f=sysdeps/unix/sysv/linux/riscv/sys/user.h
	 */
//	pc = (caddr_t)sc->sc_regs->pc;
# else
#  error "Don't know how to get the PC for this architecture in Linux!"
# endif
#elif defined(OS_FREEBSD)
# if defined(ARCH_IA32) && !defined(ARCH_IA32_x64)
	pc = (caddr_t)(uc->uc_mcontext.mc_eip);
# elif defined (ARCH_IA32) && defined(ARCH_IA32_x64)
	pc = (caddr_t)(uc->uc_mcontext.mc_rip);
# else
#  error "Don't know how to get the PC for this architecture in FreeBSD!"
# endif
#elif defined(OS_DARWIN)
# if defined(ARCH_IA32) && !defined(ARCH_IA32_x64)
	pc = (caddr_t)((uc->uc_mcontext)->__ss.__eip);
# elif defined (ARCH_IA32) && defined(ARCH_IA32_x64)
	pc = (caddr_t)((uc->uc_mcontext)->__ss.__rip);
# else
#  error "Don't know how to get the PC for this architecture in Darwin!"
# endif
#else
# error "Don't know how to get the PC for this OS!"
#endif

	Extrae_SamplingHandler ((void*) pc);

	PrepareNextAlarm ();
}


void setTimeSampling (unsigned long long period, unsigned long long variability, int sampling_type)
{
	int signum;
	int ret;

	xmemset (&signalaction, 0, sizeof(signalaction));

	ret = sigemptyset(&signalaction.sa_mask);
	if (ret != 0)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Sampling error: %s\n", strerror(ret));
		return;
	}

	if (sampling_type == SAMPLING_TIMING_VIRTUAL)
	{
		SamplingClockType = ITIMER_VIRTUAL;
		signum = SIGVTALRM;
	}
	else if (sampling_type == SAMPLING_TIMING_PROF)
	{
		SamplingClockType = ITIMER_PROF;
		signum = SIGPROF;
	}
	else
	{
		SamplingClockType = ITIMER_REAL;
		signum = SIGALRM;
	}

	ret = sigaddset(&signalaction.sa_mask, signum);
	if (ret != 0)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Sampling error: %s\n", strerror(ret));
		return;
	}

	if (variability > period)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Sampling variability can't be higher than sampling period\n");
		variability = 0;
	}

	/* The period and variability are given in nanoseconds */
	period = (period - variability) / 1000; /* We well afterwards add the variability, this is the base */
	variability = variability / 1000;
 
	SamplingPeriod_base.it_interval.tv_sec = 0;
	SamplingPeriod_base.it_interval.tv_usec = 0;
	SamplingPeriod_base.it_value.tv_sec = period / 1000000;
	SamplingPeriod_base.it_value.tv_usec = period % 1000000;

	signalaction.sa_sigaction = TimeSamplingHandler;
	signalaction.sa_flags = SA_SIGINFO | SA_RESTART;

	ret = sigaction (signum, &signalaction, NULL);
	if (ret != 0)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Sampling error: %s\n", strerror(ret));
		return;
	}

	if (variability >= RAND_MAX)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Sampling variability is too high (%llu microseconds). Setting to %llu microseconds.\n", variability, (unsigned long long) RAND_MAX);
		Sampling_variability = RAND_MAX;
	}
	else
		Sampling_variability = 2*variability;

	SamplingRunning = TRUE;

	PrepareNextAlarm ();
}


void setTimeSampling_postfork (void)
{
	int signum;
	int ret;

	if (Extrae_isSamplingEnabled())
	{
		xmemset (&signalaction, 0, sizeof(signalaction));

		ret = sigemptyset(&signalaction.sa_mask);
		if (ret != 0)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Sampling error: %s\n", strerror(ret));
			return;
		}

		if (SamplingClockType == ITIMER_VIRTUAL)
			signum = SIGVTALRM;
		else if (SamplingClockType == ITIMER_PROF)
			signum = SIGPROF;
		else
			signum = SIGALRM;

		ret = sigaddset(&signalaction.sa_mask, signum);
		if (ret != 0)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Sampling error: %s\n", strerror(ret));
			return;
		}
	
		signalaction.sa_sigaction = TimeSamplingHandler;
		signalaction.sa_flags = SA_SIGINFO | SA_RESTART;
	
		ret = sigaction (signum, &signalaction, NULL);
		if (ret != 0)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Sampling error: %s\n", strerror(ret));
			return;
		}

		SamplingRunning = TRUE;
	
		PrepareNextAlarm ();
	}
}

void unsetTimeSampling (void)
{
	if (SamplingRunning)
	{
		int ret, signum;

		if (SamplingClockType == ITIMER_VIRTUAL)
			signum = SIGVTALRM;
		else if (SamplingClockType == ITIMER_PROF)
			signum = SIGPROF;
		else
			signum = SIGALRM;

		ret = sigdelset (&signalaction.sa_mask, signum);
		if (ret != 0)
			fprintf (stderr, PACKAGE_NAME": Error Sampling error: %s\n", strerror(ret));

		SamplingRunning = FALSE;
	}
}
#else
#include <bsp.h> /* for device driver prototypes */
#include <bsp/l4stat.h>

rtems_id sampling_timer;
int sampling_period;

void sample()
{
		Backend_setInSampling(THREADID, TRUE);
		for (unsigned int thread_id = 0; thread_id < Extrae_get_num_threads(); thread_id++)
	{
			event_t evt;
			evt.time = TIME;
			evt.event = SAMPLING_EV;
			evt.value = 1;
#if defined (PAPI_COUNTERS)
			HARDWARE_COUNTERS_READ_SAMPLING(thread_id, evt, TRUE);
#endif
			BUFFER_INSERT(thread_id, SAMPLING_BUFFER(thread_id), evt);		
	}
		Backend_setInSampling(THREADID, FALSE);
}

rtems_timer_service_routine hwc_sampling(
	rtems_id timer,
	void *arg)
{
	sample();
	rtems_timer_fire_after(timer, sampling_period, hwc_sampling, NULL);
}

void setTimeSampling (int sampling_p)
{
	sampling_period=sampling_p;
	rtems_timer_create(1, &sampling_timer);
	rtems_timer_fire_after(sampling_timer, sampling_period, hwc_sampling, NULL);
}

void unsetTimeSampling (void)
{
    rtems_timer_cancel(sampling_timer);
}
void setTimeSampling_postfork (void){};
#endif

