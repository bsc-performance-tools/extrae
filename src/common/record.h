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

#ifndef RECORD_H_INCLUDED
#define RECORD_H_INCLUDED

#include "common.h"
#include "num_hwc.h"

#if defined(DEAD_CODE)
# include "clock.h"
# include "hard_counters.h"
#endif

typedef struct omp_param_t
{
  unsigned long long param;
} omp_param_t;

typedef struct misc_param_t
{
  unsigned long long param;
} misc_param_t;


typedef struct mpi_param_t
{
  int target;                   /* receiver in send - sender in receive */
  int size;
  int tag;
  int comm;
  long long aux;
} mpi_param_t;


typedef union
{
  struct omp_param_t omp_param;
  struct mpi_param_t mpi_param;
  struct misc_param_t misc_param;
} u_param;

/* HSG

  This struct contains the elements of every event that must be recorded.
  The fields must be placed in a such way that the sizeof(event_t) must
  be minimal. Each architecture has it's own preference on the alignament,
  so we must care about the packing of the structure. This is very important
  in the heterogeneous environments.
*/

typedef struct
{
  u_param param;                 /* Parameters of this event              */
  UINT64 value;                  /* Value of this event                   */
  UINT64 time;                   /* Timestamp of this event               */
#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
  long long HWCValues[MAX_HWC];  /* Hardware counters read for this event */
#endif
  int event;                     /* Type of this event                    */
#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
  int HWCReadSet;                /* Marks which set of counters was read, if any */
#endif
} event_t;


#define EVT_SIZE  sizeof(event_t)

#define Get_EvTime(ptr)          ((ptr)->time)
#define Get_EvEvent(ptr)         ((ptr)->event)
#define Get_EvValue(ptr)         ((ptr)->value)
#define Get_EvTarget(ptr)        ((ptr)->param.mpi_param.target)
#define Get_EvSize(ptr)          ((ptr)->param.mpi_param.size)
#define Get_EvTag(ptr)           ((ptr)->param.mpi_param.tag)
#define Get_EvComm(ptr)          ((ptr)->param.mpi_param.comm)
#define Get_EvAux(ptr)           ((ptr)->param.mpi_param.aux)
#define Get_EvParam(ptr)         ((ptr)->param.omp_param.param)
#define Get_EvMiscParam(ptr)     ((ptr)->param.misc_param.param)
#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
# define Get_EvHWCRead(ptr)      (((ptr)->HWCReadSet > 0) ? 1 : 0) /* 0 = not read, >0 = set_id + 1 */
# define Get_EvHWCSet(ptr)       ((ptr)->HWCReadSet - 1) 
# define Get_EvHWCVal(ptr)       ((ptr)->HWCValues)
#endif /* USE_HARDWARE_COUNTERS */

#endif /* RECORD_H_INCLUDED */

