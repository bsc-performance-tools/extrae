#ifndef gm_klog_debug_h
#define gm_klog_debug_h

#if 0	/* log event, this is not really Linux specific  */

#define GM_DEBUG_KLOG 0
#define GM_KLOG_EVT(evt,arg) log_evt(evt,arg)

#define GM_KLOG_NBEVT 2048
enum gm_klog_evt
{
  GM_KLOG_INTR, GM_KLOG_SEND, GM_KLOG_BROAD, GM_KLOG_RECV, GM_KLOG_PROVIDE,
  GM_KLOG_SFREE, GM_KLOG_SENDEND, GM_KLOG_SQUEUE, GM_KLOG_SQUEUEND,
  GM_KLOG_SENDSEG, GM_KLOG_SENDSEGEND, GM_KLOG_INTR0
};

struct gm_klog_struct
{
  unsigned long rpcc;
  void *addr;
  enum gm_klog_evt type;
};

extern struct gm_klog_struct gm_klog_tab[AC_NBEVT];
extern atomic_t gm_klog_index;

static
gm_klog_evt (enum ac_evt type, void *arg)
{
  long index = (atomic_inc_return (&ac_index) - 1) % AC_NBEVT;
  ac_tab[index].type = type;
#ifdef __alpha__
  __asm__ volatile ("rpcc %0":"=r" (ac_tab[index].rpcc));
#endif
  ac_tab[index].addr = arg;
}
#else
#define GM_KLOG_EVT(evt,arg)
#endif



#endif /* gm_klog_debug_h */
