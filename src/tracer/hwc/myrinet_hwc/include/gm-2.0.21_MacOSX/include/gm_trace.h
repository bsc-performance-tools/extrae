#ifndef gm_trace_t
#define gm_trace_t

#include "gm_enable_trace.h"

typedef gm_u64_t gm_stamp_t;


typedef gm_u32_n_t gm_l_stamp_t;


typedef struct
{
  gm_l_stamp_t stamp;
  gm_u32_n_t evt;
} gm_l_trace_t;



typedef struct gm_file_trace
{
  gm_stamp_t stamp;
  gm_u16_t evt;
  gm_u16_t aux1;
  gm_u32_t aux2;
} gm_file_trace_t;

enum host_event {
  GM_SEND_EVENT = 256,
  GM_AFTER_SEND_EVENT,
  GM_RECEIVE_EVENT,
  GM_AFTER_RECEIVE_EVENT,
  GM_AFTER_PAGE_TRANSLATION,
  GM_GOT_CHUNK,
  GM_ETHER_SND,
  GM_ETHER_RCV,
  GM_ETHER_SINTR,
  GM_SET_DMA_CTR,
  GM_UPDATE_HSQ
};

struct gm_file_trace;
GM_ENTRY_POINT gm_status_t gm_get_ktrace (struct gm_port *,
					  struct gm_file_trace *);
GM_ENTRY_POINT void gm_save_trace(struct gm_port *gm_port);
GM_ENTRY_POINT void gm_tracing (struct gm_port *,int enable);

	

#define GM_TRACEBUFSIZE (GM_LANAI_NUMTRACE*8/*sizeof(gm_ltrace_t)*/)

#if GM_ENABLE_TRACE

#if GM_TRACEBUFSIZE > 32768
#error assembly "and" may not work, it may requires manual modification, but maybe not...
#endif

#define GM_LANAI_TRACEMASK (~(GM_TRACEBUFSIZE))

#if GM_MCP

#define GM_LOG_EVT(_evt) do {						\
  G_TRACEPTR &= GM_LANAI_TRACEMASK;					\
  ((unsigned*)G_TRACEPTR)++[0] = RTC;					\
  ((unsigned*)G_TRACEPTR)++[0] = _evt;					\
} while (0)

#else

#include "gm_tick.h"


#if GM_KERNEL
/* kernel definition must be reentrant (interrupt handling), we need atomic counting operations */
#include "gm_arch_def.h"
#define GM_ATOMIC_INC_AND_RET(i) gm_arch_atomic_preinc(&i)
#define GM_ATOMIC_SET(i,v) gm_arch_atomic_set(&(i),(v))
#define GM_ATOMIC_READ(i) gm_arch_atomic_read(&(i))
#else
#define GM_ATOMIC_INC_AND_RET(i) (++(i))
#define GM_ATOMIC_SET(i,v) ((i) = (v))
#define GM_ATOMIC_READ(i) (i)
typedef unsigned  gm_atomic_t;
#endif

/* the first event is at index 1, 0 means off */
extern gm_file_trace_t gm_trace_log[];
extern gm_atomic_t gm_trace_index;

/* we switch off tracing incase of host overflow */
#define GM_LOG_STAMPED_EVT(event,tick,arg1,arg2)	do {		\
  if (GM_ATOMIC_READ(gm_trace_index) >= 1) {				\
     unsigned _index = GM_ATOMIC_INC_AND_RET(gm_trace_index) - 1;	\
     if (_index >= GM_HOST_NUMTRACE - 1)				\
        GM_ATOMIC_SET(gm_trace_index,0);				\
     else {								\
       gm_trace_log[_index].stamp = (tick);				\
       gm_trace_log[_index].evt = (event);				\
       gm_trace_log[_index].aux1 = (arg1);				\
       gm_trace_log[_index].aux2 = (arg2);				\
     }									\
  } } while (0)
#define GM_LOG_EVT(evt) GM_LOG_STAMPED_EVT(evt,gm_tick(),0,0)
#endif /* GM_MCP or HOST */

#else /* !GM_ENABLE_TRACE  */

#define GM_LOG_EVT(evt)
#define GM_LOG_STAMPED_EVT(evt,tick,a0,a1)
#define GM_LANAI_TRACEMASK 0

#endif /* !GM_ENABLE_TRACE */


#endif

