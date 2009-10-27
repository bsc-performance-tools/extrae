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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/spu/wrapper.h,v $
 | 
 | @last_commit: $Date: 2009/06/19 14:30:26 $
 | @version:     $Revision: 1.3 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __WRAPPER_H__
#define __WRAPPER_H__

#include <record.h>
#include "spu_clock.h"

/************ Structs **************/

#define FD(i)         (PRDAUSR[i].fd)
#define CUREVT(i)     (PRDAUSR[i].cur)
#define LASTEVT(i)    (PRDAUSR[i].last)
#define FIRSTEVT(i)   (PRDAUSR[i].first)
#define HEADEVT(i)    (PRDAUSR[i].head)
#define FLUSHED(i)    (PRDAUSR[i].flushed)
#define RANK(i)         (PRDAUSR[i].rank)
#define PRDAVPID(i)     (PRDAUSR[i].whoami)
#define HWCEVTSET(i)    (PRDAUSR[i].HWCEventSet)

struct trace_prda
{
  event_t *cur;                 /* 8 bytes */
  event_t *last;                /* 8 bytes */
  event_t *first;               /* 8 bytes */
  event_t *head;                /* 8 bytes */
  int fd;                       /* 4 bytes */
  unsigned int rank;            /* 4 bytes */
  unsigned int whoami;          /* 4 bytes */
  unsigned int flushed;         /* 4 bytes */
};
/* Total size : 80 bytes (base) */

/************ Function prototypes **************/

void flush_buffer (int mark_on_trace, int thread);
int spu_init_backend (int me, unsigned long long trace_ptr, unsigned long long count_trace_ptr, unsigned int file_size);
void Thread_Finalization ();
void advance_current(int thread);
void Touch_PPU_Buffer (void);

/************ Global variables that control whether tracing is enabled **************/

extern int tracejant;      /* Enables to stop tracing the whole application */
extern int *TracingBitmap; /* Enables to trace a subset of tasks */

extern struct trace_prda *PRDAUSR;

/************ Tracing macros **************/

#define CELLTRACE_EVENT(evttime,evttype,evtvalue) \
{                                                 \
  int __thread_id__ = THREADID;                   \
  if (tracejant)                                  \
  {                                               \
    CUREVT(__thread_id__)->time =  evttime;       \
    CUREVT(__thread_id__)->event = evttype;       \
    CUREVT(__thread_id__)->value = evtvalue;      \
    advance_current(__thread_id__);               \
  }                                               \
}

#define CELLTRACE_MISCEVENT(evttime,evttype,evtvalue,evtparam)                         \
{                                                                                      \
  int __thread_id__ = THREADID;                                                        \
  if (tracejant)                                                                       \
  {                                                                                    \
    CUREVT(__thread_id__)->time = (evttime);                                           \
    CUREVT(__thread_id__)->event = (evttype);                                          \
    CUREVT(__thread_id__)->value = (evtvalue);                                         \
    CUREVT(__thread_id__)->param.misc_param.param = (unsigned long long) (evtparam);   \
    advance_current(__thread_id__);                                                    \
  }                                                                                    \
}

#endif /* __WRAPPER_H__ */
