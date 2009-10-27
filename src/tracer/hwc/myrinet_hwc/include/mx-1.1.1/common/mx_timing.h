/*************************************************************************
 * The contents of this file are subject to the MYRICOM MYRINET          *
 * EXPRESS (MX) NETWORKING SOFTWARE AND DOCUMENTATION LICENSE (the       *
 * "License"); User may not use this file except in compliance with the  *
 * License.  The full text of the License can found in LICENSE.TXT       *
 *                                                                       *
 * Software distributed under the License is distributed on an "AS IS"   *
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied.  See  *
 * the License for the specific language governing rights and            *
 * limitations under the License.                                        *
 *                                                                       *
 * Copyright 2005 by Myricom, Inc.  All rights reserved.                 *
 *************************************************************************/

#ifndef _mx_timing_h
#define _mx_timing_h

#include "mx_auto_config.h"
#include "myriexpress.h"

typedef uint64_t mx_cycles_t;

#if !MX_OS_WINNT
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#endif


#ifdef __GNUC__
#if MX_CPU_x86  || MX_CPU_x86_64

static inline mx_cycles_t mx_get_cycles(void)
{
  unsigned l,h;
  asm volatile("rdtsc": "=a" (l), "=d" (h));
  return l + (((mx_cycles_t)h) << 32);
}

#elif MX_CPU_powerpc || MX_CPU_powerpc64

static inline mx_cycles_t mx_get_cycles()
		 /*+ The number of cycles elapsed. +*/ 
{ 
  mx_cycles_t t;
  unsigned tbu, tbl, tbu2;
  if (sizeof(void *) == 8) {
    asm volatile("mftb %0" : "=r" (t)); 
    return t;
  } else {
    do {
      asm volatile ("mftbu %0" : "=r" (tbu));
      asm volatile ("mftb %0" : "=r" (tbl));
      asm volatile ("mftbu %0" : "=r" (tbu2));
    } while (tbu != tbu2);
    return ((mx_cycles_t)tbu << 32) + tbl;
  }
}

#elif MX_CPU_ia64

static inline mx_cycles_t mx_get_cycles()
{ 
  mx_cycles_t t;
  asm volatile ("mov %0=ar.itc" : "=r"(t));
  return t;
}

#elif MX_CPU_alpha
static inline mx_cycles_t mx_get_cycles()
		 /*+ The number of cycles elapsed. +*/ 
{ 
   mx_cycles_t t;
   asm volatile ("rpcc %0" : "=r" (t));
   /* according to the brown book, (I) 4-143, the lower
    * 32-bits are an unsigned, wrapping counter, but the
    * upper-32-bits are OS dependant.  So just use the
    * lower 32-bits */
   return t & 0xffffffffLL; 
}

#else
#error mx_get_cycles not implemented
#endif

double mx_seconds_per_cycle(void);

mx_cycles_t mx_cycles_per_second(void);

#else /* GNU_C compiler */

#if !MX_OS_WINNT

#include <unistd.h>
#include <sys/time.h>
static inline mx_cycles_t mx_get_cycles() 
{
  struct timeval t;
  gettimeofday(&t,NULL);
  return (mx_cycles_t)t.tv_sec*1000000+t.tv_usec;
}

#define mx_seconds_per_cycle() 1e-6
#define mx_cycles_per_second() ((mx_cycles_t)1000000)

#else /* !MX_OS_WINNT */

double mx_seconds_per_cycle(void);

#if MX_CPU_x86
#include <windows.h>
static inline mx_cycles_t mx_get_cycles(void)
{
  LARGE_INTEGER x;
  __asm {
    _emit 0fh;
    _emit 31h;
    mov x.HighPart, edx;
    mov x.LowPart, eax;
  }
  return x.QuadPart;
}
#elif MX_CPU_x86_64
#define mx_get_cycles __rdtsc
#endif

#endif

#endif


void mx_cycles_counter_init(void);


#endif /* _mx_timing_h */
