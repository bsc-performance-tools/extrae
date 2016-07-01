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

#if defined(IS_BGQ_MACHINE)

#include <firmware/include/personality.h>
#include <spi/include/kernel/process.h>
#include <spi/include/kernel/location.h>

#ifdef __GNUC__
#include <hwi/include/bqc/A2_inlines.h>
#endif

#ifdef HAVE_RTS_H
# include <rts.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#include "bgq_clock.h" 

#define BITS_FOR_CYCLES_FACTOR 	14
#define CYCLES_FACTOR          	(1ULL<<BITS_FOR_CYCLES_FACTOR)

static unsigned long long factor;

/* From bpcore/ppc450_inlines.h */

unsigned long long  _bgq_GetTimeBase( void )
{
#if defined (__GNUC__)
	return GetTimeBase();
#elif defined (__IBMC__)
	return __mftb();
#else
	#error "Cannot find GetTimeBase for BG/Q (unhandled compiler)"
#endif
}

#define BGQ_DEFAULT_CPU_FREQ_MHZ 1600ULL
void bgq_Initialize (void)
{
	unsigned long long freqMHz = BGQ_DEFAULT_CPU_FREQ_MHZ; /* use a correct default */

#if defined (__GNUC__)
	Personality_t personality;
	unsigned personality_size = sizeof (personality);

	Kernel_GetPersonality(&personality, personality_size);
	freqMHz = personality.Kernel_Config.FreqMHz;
#endif

	factor = (1000ULL * CYCLES_FACTOR / freqMHz) + 1;
}

iotimer_t bgq_getTime (void)
{
	return (_bgq_GetTimeBase() * factor) >> BITS_FOR_CYCLES_FACTOR;
}

#endif /* IS_BGQ_MACHINE */
