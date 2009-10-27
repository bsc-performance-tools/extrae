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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/hwc/myrinet_hwc.h,v $
 | 
 | @last_commit: $Date: 2007/09/21 16:33:39 $
 | @version:     $Revision: 1.1.1.1 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __MYRINET_HWC_H__
#define __MYRINET_HWC_H__

#define MYRINET_BASE_EV    44000000
#define ROUTES_BASE_EV 110000001

typedef enum {
    GM,
    MX
} driver_t;

#ifdef HAVE_STDINT_H
# include <stdint.h>
#endif

extern int Myrinet_Counters_Enabled;
extern int Myrinet_Counters_Count;
extern int Myrinet_Routes_Enabled;
extern int Myrinet_Routes_Count;
extern driver_t Myrinet_Driver;

extern int (*Myrinet_read_counters) (int, uint32_t *);
extern int (*Myrinet_read_routes)   (int, int, uint32_t *);

void Myrinet_HWC_Initialize(void);

#define TRACE_MYRINET_HWC()                                         \
{                                                                   \
   if (Myrinet_Counters_Enabled)                                    \
   {                                                                \
      int i;                                                        \
      uint32_t types  [Myrinet_Counters_Count];                     \
      uint32_t values [Myrinet_Counters_Count];                     \
      uint32_t params [Myrinet_Counters_Count];                     \
      iotimer_t timestamp = TIME;                                   \
                                                                    \
      (*Myrinet_read_counters)(Myrinet_Counters_Count, params);     \
      for (i=0; i<Myrinet_Counters_Count; i++)                      \
      {                                                             \
         /* Would be nice to do this just once */                   \
         types [i] = USER_EV;                                       \
         values[i] = MYRINET_BASE_EV + (Myrinet_Driver*100000) + i; \
      }                                                             \
      TRACE_N_MISCEVENT(timestamp, Myrinet_Counters_Count,          \
                        types, values, params);                     \
   }                                                                \
}

#define TRACE_MYRINET_ROUTES(mpi_rank)                                \
{                                                                     \
   if ((Myrinet_Counters_Enabled) && (Myrinet_Routes_Enabled))        \
   {                                                                  \
      int i;                                                          \
      uint32_t types  [Myrinet_Routes_Count];                         \
      uint32_t values [Myrinet_Routes_Count];                         \
      uint32_t params [Myrinet_Routes_Count];                         \
      iotimer_t timestamp = TIME;                                     \
                                                                      \
      (*Myrinet_read_routes)(mpi_rank, Myrinet_Routes_Count, params); \
                                                                      \
      for (i=0; i<Myrinet_Routes_Count; i++)                          \
      {                                                               \
         /* Would be nice to do this just once */                     \
         types [i] = USER_EV;                                         \
         values[i] = ROUTES_BASE_EV + (mpi_rank * 1000) + i;          \
      }                                                               \
      TRACE_N_MISCEVENT(timestamp, Myrinet_Counters_Count,            \
                        types, values, params);                       \
	}                                                                 \
}

#endif /* __MYRINET_HWC_H__ */
