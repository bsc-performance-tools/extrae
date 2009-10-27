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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/paraver/misc_prv_semantics.h,v $
 | 
 | @last_commit: $Date: 2008/01/21 08:49:07 $
 | @version:     $Revision: 1.2 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __MISC_PRV_SEMANTICS_H__
#define __MISC_PRV_SEMANTICS_H__

#include "record.h"
#include "semantics.h"
#include "file_set.h"

extern int MPI_Caller_Multiple_Levels_Traced;
extern int *MPI_Caller_Labels_Used;

extern int Sample_Caller_Multiple_Levels_Traced;
extern int *Sample_Caller_Labels_Used;

extern SingleEv_Handler_t PRV_MISC_Event_Handlers[];
extern RangeEv_Handler_t PRV_MISC_Range_Handlers[];


#endif /* __MISC_PRV_SEMANTICS_H__ */
