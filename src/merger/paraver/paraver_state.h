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

#ifndef PARAVER_STATE_H
#define PARAVER_STATE_H

#include "file_set.h"

unsigned int Push_State (unsigned int new_state, unsigned int ptask, unsigned int task, unsigned int thread);
unsigned int Pop_State (unsigned int old_state, unsigned int ptask, unsigned int task, unsigned int thread);
unsigned int Switch_State (unsigned int state, int condition, unsigned int ptask, unsigned int task, unsigned int thread);
unsigned int Top_State (unsigned int ptask, unsigned int task, unsigned int thread);
int State_Excluded (unsigned int state);
void Initialize_Trace_Mode_States (unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread, int mode);
void Initialize_States (FileSet_t * fset);
void Finalize_States (FileSet_t * fset, unsigned long long current_time);

int Get_Joint_States (void);
int Get_Last_State (void);

#endif
