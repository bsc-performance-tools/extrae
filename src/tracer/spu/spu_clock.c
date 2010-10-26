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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#include <spu_intrinsics.h>
#include <spu_internals.h>
#include "spu_clock.h"

unsigned long long timebase_MHz;
unsigned long long timeInit;

void spu_clock_init(unsigned long long timebase, unsigned long long temps) {
	timebase_MHz = timebase / 1000000;
	timeInit = temps;

  spu_writech(22, 1);
  spu_writech(MFC_WR_EVENT_MASK, 0);
  spu_writech(MFC_WR_EVENT_ACK, MFC_DECREMENTER_EVENT);
  spu_writech(MFC_WR_DECR_COUNT, -1);
}

unsigned long long get_spu_time(void) {
	static unsigned long long last_read = 0;
	unsigned long long current_read = ~(spu_readch(MFC_RD_DECR_COUNT));

	/* If the time counter has overflown, add time counter maximum to timeInit */
	if (last_read > current_read)
		timeInit += 0x100000000LL;

	last_read = current_read;

	unsigned long long result = ((current_read * 1000) / timebase_MHz) + timeInit;

	return result;
}

