/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                   Extrae                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *     ___     This library is free software; you can redistribute it and/or *
 *    /  __         modify it under the terms of the GNU LGPL as published   *
 *   /  /  _____    by the Free Software Foundation; either version 2.1      *
 *  /  /  /     \   of the License, or (at your option) any later version.   *
 * (  (  ( B S C )                                                           *
 *  \  \  \_____/   This library is distributed in hope that it will be      *
 *   \  \__         useful but WITHOUT ANY WARRANTY; without even the        *
 *    \___          implied warranty of MERCHANTABILITY or FITNESS FOR A     *
 *                  PARTICULAR PURPOSE. See the GNU LGPL for more details.   *
 *                                                                           *
 * You should have received a copy of the GNU Lesser General Public License  *
 * along with this library; if not, write to the Free Software Foundation,   *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA          *
 * The GNU LEsser General Public License is contained in the file COPYING.   *
 *                                 ---------                                 *
 *   Barcelona Supercomputing Center - Centro Nacional de Supercomputacion   *
\*****************************************************************************/

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

#if defined(TEMPORARILY_DISABLED)
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
#else 
#define TRACE_MYRINET_HWC()
#define TRACE_MYRINET_ROUTES(mpi_rank)
#endif


#endif /* __MYRINET_HWC_H__ */
