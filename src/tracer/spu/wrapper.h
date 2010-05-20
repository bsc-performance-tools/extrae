/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
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
int spu_init_backend (int me, unsigned long long trace_ptr, unsigned long long count_trace_ptr, unsigned int file_size, int fd);
void Thread_Finalization ();
void advance_current(int thread);
#ifndef SPU_USES_WRITE
void Touch_PPU_Buffer (void);
#endif

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
