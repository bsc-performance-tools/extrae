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
