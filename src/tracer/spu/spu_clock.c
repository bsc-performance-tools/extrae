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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/spu/spu_clock.c,v $
 | 
 | @last_commit: $Date: 2008/05/19 09:59:33 $
 | @version:     $Revision: 1.3 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: spu_clock.c,v 1.3 2008/05/19 09:59:33 harald Exp $";

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
	unsigned long long temps = ~(spu_readch(MFC_RD_DECR_COUNT));
	unsigned long long result = ((temps * 1000) / timebase_MHz) + timeInit;
	return result;
}

