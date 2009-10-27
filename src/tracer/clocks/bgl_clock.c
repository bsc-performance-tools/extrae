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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/clocks/bgl_clock.c,v $
 | 
 | @last_commit: $Date: 2008/09/29 09:24:12 $
 | @version:     $Revision: 1.4 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: bgl_clock.c,v 1.4 2008/09/29 09:24:12 harald Exp $";

#if defined(IS_BGL_MACHINE)

#ifdef HAVE_RTS_H
# include <rts.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#include "bgl_clock.h"

#define BITS_FOR_CYCLES_FACTOR 	14
#define CYCLES_FACTOR          	(1ULL<<BITS_FOR_CYCLES_FACTOR)

static unsigned long long factor;

#if defined(DEAD_CODE)
static inline unsigned long long timebase (void)
{
#define SPRN_TBRL 0x10C     /* Time Base Read Lower Register (user & sup R/O) */
#define SPRN_TBRU 0x10D     /* Time Base Read Upper Register (user & sup R/O) */
	unsigned volatile u1, u2, lo;
	union
	{
		struct
		{
			unsigned hi, lo;
		}
		w;
		unsigned long long d;
	}
	result;

	do
	{
		asm volatile ("mfspr %0,%1":"=r" (u1):"i" (SPRN_TBRU));
		asm volatile ("mfspr %0,%1":"=r" (lo):"i" (SPRN_TBRL));
		asm volatile ("mfspr %0,%1":"=r" (u2):"i" (SPRN_TBRU));
	}
	while (u1 != u2);

	result.w.lo = lo;
	result.w.hi = u2;
	return result.d;
}
#endif

void bgl_Initialize (void)
{
	BGLPersonality personality;
	unsigned personality_size = sizeof (personality);

	rts_get_personality (&personality, personality_size);
	factor = (1000000000ULL * CYCLES_FACTOR / personality.clockHz) + 1;
}

void bgl_Initialize_thread (void)
{
}

iotimer_t bgl_getTime (void)
{
	return (rts_get_timebase() * factor) >> BITS_FOR_CYCLES_FACTOR;
}

#endif /* IS_BGL_MACHINE */
