/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
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

#ifndef __SEMANTICS_H_INCLUDED__
#define __SEMANTICS_H_INCLUDED__

#include "file_set.h"
#include "events.h"

enum
{
	PRV_SEMANTICS,
	TRF_SEMANTICS
};

typedef int Ev_Handler_t(event_t *, unsigned long long, unsigned int, unsigned int, unsigned int, unsigned int, FileSet_t *);

typedef struct
{
	int event;
	Ev_Handler_t *handler;
} SingleEv_Handler_t;

typedef struct
{
	int range_min;
	int range_max;
	Ev_Handler_t *handler;
} RangeEv_Handler_t;

/* public: */
void Semantics_Initialize (int output_format);
Ev_Handler_t * Semantics_getEventHandler (int event);
int SkipHandler (event_t *, unsigned long long, unsigned int, unsigned int, unsigned int, unsigned int, FileSet_t *);

#endif /* __SEMANTICS_H_INCLUDED__ */
