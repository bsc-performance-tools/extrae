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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/clocks/bgp_clock.c,v $
 | 
 | @last_commit: $Date: 2009/02/09 15:12:16 $
 | @version:     $Revision: 1.2 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: bgp_clock.c,v 1.2 2009/02/09 15:12:16 harald Exp $";

#if defined(IS_BGP_MACHINE)

/* GRH: Three next includes come from the redbook example */
#include <spi/kernel_interface.h>
#include <common/bgp_personality.h>
#include <common/bgp_personality_inlines.h>

#ifdef HAVE_RTS_H
# include <rts.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#include "bgp_clock.h" 

#define BITS_FOR_CYCLES_FACTOR 	14
#define CYCLES_FACTOR          	(1ULL<<BITS_FOR_CYCLES_FACTOR)

static unsigned long long factor;

/* From bpcore/ppc450_inlines.h */

unsigned long long  _bgp_GetTimeBase( void )
{
	union
	{
		unsigned ul[2];
		unsigned long long ull;
	} hack;
	unsigned utmp;

	do {
		utmp       = _bgp_mfspr( SPRN_TBRU );
		hack.ul[1] = _bgp_mfspr( SPRN_TBRL );
		hack.ul[0] = _bgp_mfspr( SPRN_TBRU );
	}
	while (utmp != hack.ul[0]);

	return hack.ull;
}

void bgp_Initialize (void)
{
	_BGP_Personality_t personality;
	unsigned personality_size = sizeof (personality);

	Kernel_GetPersonality(&personality, personality_size);
	factor = (1000000000ULL * CYCLES_FACTOR / (1000000ULL * BGP_Personality_clockMHz(&personality))) + 1;
}

void bgp_Initialize_thread (void)
{
}

iotimer_t bgp_getTime (void)
{
	return (_bgp_GetTimeBase() * factor) >> BITS_FOR_CYCLES_FACTOR;
}

#endif /* IS_BGP_MACHINE */
