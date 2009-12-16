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

#ifndef MISC_PRV_EVENTS_H
#define MISC_PRV_EVENTS_H

#if HAVE_STDIO_H
# include <stdio.h>
#endif

void MISCEvent_WriteEnabledOperations (FILE * fd, long long options);

#define MN_LINEAR_HOST_LABEL  "Linear host number"
#define MN_LINECARD_LABEL     "Linecard"
#define MN_HOST_LABEL         "Node inside linecard"

#define BG_TORUS_X            "BG X Coordinate in Torus"
#define BG_TORUS_Y            "BG Y Coordinate in Torus"
#define BG_TORUS_Z            "BG Z Coordinate in Torus"
#define BG_PROCESSOR_ID       "BG Processor ID"

#if defined(PARALLEL_MERGE)
void Share_MISC_Operations (void);
#endif

#endif
