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

#ifndef SPU_INCLUDED_H
#define SPU_INCLUDED_H

#ifdef __cplusplus
extern "C" {
#endif

int SPUtrace_init (void);
void SPUtrace_fini (void);
void SPUtrace_event (unsigned int type, unsigned int value);
void SPUtrace_shutdown (void);
void SPUtrace_restart (void);

int MPItrace_init (void) __attribute__ ((deprecated));
void MPItrace_fini (void) __attribute__ ((deprecated));
void MPItrace_event (unsigned int type, unsigned int value) __attribute__ ((deprecated));
void MPItrace_shutdown (void) __attribute__ ((deprecated));
void MPItrace_restart (void) __attribute__ ((deprecated));

#ifdef __cplusplus
}
#endif

#endif /* PPU_INCLUDED_H */
