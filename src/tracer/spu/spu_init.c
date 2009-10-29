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
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
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

int SPUtrace_init (void) {
	unsigned long long timebase, timeinit;
	unsigned long long prvout, countout;
	unsigned int prvout_high, prvout_low, countout_high, countout_low;
	unsigned int TB_high, TB_low, spu_creation_time_high, spu_creation_time_low, spu_buffer_size, spu_file_size, dma_channel;
	int ret, me;

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

	ret = spu_init_backend (me, prvout, countout, spu_file_size);

	Touch_PPU_Buffer();

	/* Tell the PPU how the init performed */
	spu_write_out_mbox (ret);

	if (!ret)
		exit (-1);

	return 1;
}
int MPItrace_init (void) __attribute__ ((alias ("SPUtrace_init")));
int MPItrace_init (void) __attribute__ ((deprecated));


/******************************************************************************
 ***  MPI_Finalize_C_Wrapper
 ******************************************************************************/
int SPUtrace_fini (void)
{
	if (mpitrace_on)
		Thread_Finalization ();
	return 0;
}
int MPItrace_fini (void) __attribute__ ((alias ("SPUtrace_fini")));
int MPItrace_fini (void) __attribute__ ((deprecated));
