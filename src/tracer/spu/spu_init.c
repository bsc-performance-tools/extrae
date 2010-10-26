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

#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <spu_intrinsics.h>
#include <spu_mfcio.h>
#include "spu_clock.h"
#include "wrapper.h"

#ifdef SPU_DYNAMIC_BUFFER
extern int EVENT_BUFFER_SIZE;
#endif

extern int mpitrace_on;

int SPUtrace_init (void) __attribute__ ((alias ("Extrae_init"))); 
int Extrae_init (void)
{
	unsigned long long timebase, timeinit;
	unsigned long long prvout, countout;
	unsigned int prvout_high, prvout_low, countout_high, countout_low;
	unsigned int TB_high, TB_low, spu_creation_time_high, spu_creation_time_low, spu_buffer_size, spu_file_size, dma_channel;
	int ret, me;
	int prv_fd;

	/* Is the tracing enabled? If not, just leave!*/
	mpitrace_on = spu_read_in_mbox();
	if (!mpitrace_on)
		return 0;

	TB_high = spu_read_in_mbox();
	TB_low = spu_read_in_mbox();
	timebase = ((unsigned long long)TB_high << 32) | TB_low;

	spu_creation_time_high = spu_read_in_mbox();
	spu_creation_time_low = spu_read_in_mbox();
	timeinit = ((unsigned long long)spu_creation_time_high << 32) | spu_creation_time_low;

	spu_clock_init(timebase, timeinit);

	me = spu_read_in_mbox();
	spu_file_size = spu_read_in_mbox();

	prv_fd = spu_read_in_mbox ();

	prvout_high = spu_read_in_mbox();
	prvout_low = spu_read_in_mbox();
	prvout = ((unsigned long long)prvout_high << 32) | prvout_low;

	countout_high = spu_read_in_mbox();
	countout_low = spu_read_in_mbox();
	countout = ((unsigned long long)countout_high << 32) | countout_low;

	dma_channel = spu_read_in_mbox();

	spu_buffer_size = spu_read_in_mbox();

#ifdef SPU_DYNAMIC_BUFFER
	EVENT_BUFFER_SIZE = spu_buffer_size;
#endif

	ret = spu_init_backend (me, prvout, countout, spu_file_size, prv_fd);

#ifdef SPU_USES_WRITE
#else
	Touch_PPU_Buffer();
#endif

	/* Tell the PPU how the init performed */
	spu_write_out_mbox (ret);

	if (!ret)
		exit (-1);

	return 1;
}


/******************************************************************************
 ***  MPI_Finalize_C_Wrapper
 ******************************************************************************/
int SPUtrace_fini (void) __attribute__ ((alias ("Extrae_fini"))); 
int Extrae_fini (void)
{
	if (mpitrace_on)
		Thread_Finalization ();
	return 0;
}
