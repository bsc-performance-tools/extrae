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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/clocks/clock.h,v $
 | 
 | @last_commit: $Date: 2009/05/28 13:38:53 $
 | @version:     $Revision: 1.6 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef CLOCK_H
#define CLOCK_H

enum
{
  REAL_CLOCK = 0,
  USER_CLOCK,
};

/* 
 * In Cell/64, the PPU is compiled in 64-bit, but the SPU is always compiled in 32-bit.
 * We need iotimer_t to be mapped into a 64-bit type both in the PPU and SPU, so we try to use unsigned long long first.
 */
#if SIZEOF_LONG_LONG == 8 
typedef unsigned long long iotimer_t;
#elif SIZEOF_LONG == 8
typedef unsigned long iotimer_t;
#endif

#define TIME (Clock_getTime())
#define CLOCK_INIT (Clock_Initialize())
#define CLOCK_INIT_THREAD (Clock_Initialize())

#if defined(__cplusplus)
extern "C" {
#endif
void Clock_setType (unsigned type);
unsigned Clock_getType (void);

iotimer_t Clock_getTime (void);
void Clock_Initialize (void);
void Clock_Initialize_thread (void);
#if defined(__cplusplus)
}
#endif

#endif
