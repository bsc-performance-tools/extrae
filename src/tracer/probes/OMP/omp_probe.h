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

#ifndef OMP_PROBE_H_INCLUDED
#define OMP_PROBE_H_INCLUDED

void Probe_OpenMP_UF (UINT64 uf);
void Probe_OpenMP_Work_Entry (void);
void Probe_OpenMP_Work_Exit (void);

void Probe_OpenMP_Join_NoWait_Entry (void);
void Probe_OpenMP_Join_NoWait_Exit (void);
void Probe_OpenMP_Join_Wait_Entry (void);
void Probe_OpenMP_Join_Wait_Exit (void);

void Probe_OpenMP_DO_Entry (void);
void Probe_OpenMP_DO_Exit (void);
void Probe_OpenMP_Sections_Entry (void);
void Probe_OpenMP_Sections_Entry (void);
void Probe_OpenMP_Single_Entry (void);
void Probe_OpenMP_Single_Exit (void);
void Probe_OpenMP_Section_Entry(void);
void Probe_OpenMP_Section_Exit (void);

void Probe_OpenMP_ParRegion_Entry (void);
void Probe_OpenMP_ParRegion_Exit (void);
void Probe_OpenMP_ParDO_Entry (void);
void Probe_OpenMP_ParDO_Exit (void);
void Probe_OpenMP_ParSections_Entry (void);
void Probe_OpenMP_ParSections_Exit (void);

void Probe_OpenMP_Barrier_Entry (void);
void Probe_OpenMP_Barrier_Exit (void);

void Probe_OpenMP_Named_Lock_Entry (void);
void Probe_OpenMP_Named_Lock_Exit (void);
void Probe_OpenMP_Named_Unlock_Entry (void);
void Probe_OpenMP_Named_Unlock_Exit (void);
void Probe_OpenMP_Unnamed_Lock_Entry (void);
void Probe_OpenMP_Unnamed_Lock_Exit (void);
void Probe_OpenMP_Unnamed_Unlock_Entry (void);
void Probe_OpenMP_Unnamed_Unlock_Exit (void);

void setTrace_OMPLocks (int value);

#endif
