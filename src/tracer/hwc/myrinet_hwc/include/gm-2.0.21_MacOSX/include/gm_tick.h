#ifndef _gm_tick_h_
#define  _gm_tick_h_

#include "gm_enable_trace.h"
#if GM_CPU_mips
#include <time.h>
#endif

#if GM_ENABLE_TRACE

typedef gm_u64_t gm_tick_t;

void gm_tick_init(void);
gm_tick_t gm_tick_recompute(void);

static gm_inline gm_tick_t gm_tick(void)
{
#if GM_CPU_x86 && defined __GNUC__
  unsigned lsw,msw;
  __asm__ volatile("rdtsc" : "=a" (lsw), "=d" (msw));
  return lsw + (((gm_tick_t)msw)<<32);
#elif GM_CPU_x86 && defined _MSC_VER
#if 0
  /* aggressive version
     rdtsc leaves its result in edx,eax
     leave out the return call and rely on the fact that return value is
     expected to be in edx,eax
     the compiler might complain about missing return call */
  _asm {
    _emit 0x0f;
    _emit 0x31;
  }
#else
  /* paranoid version */
  union {
    unsigned long x[2];
    unsigned _int64 y;
  } u;
  _asm {
    _emit 0x0f;
    _emit 0x31;
    mov u.x[0], edx;
    mov u.x[1], eax;
  }
  return u.y;
#endif
#elif GM_CPU_alpha && defined __GNUC__
  unsigned tick;
  extern unsigned gm_tick_high, gm_tick_last;
  __asm__ volatile("rpcc %0" : "=r" (tick));
  /* reserve 8 bits for statically good recovery in case of long
     sleep: 1 failure out of 256 */
  if (tick <= gm_tick_last
      || (unsigned)(tick - gm_tick_last) > (1U << (32-8)))
    {
      return gm_tick_recompute();
    }
  else
    {
      gm_tick_last = tick;
      return tick+((unsigned long)gm_tick_high<<32);
    }
#elif GM_CPU_powerpc && defined __GNUC__
  unsigned lsw,msw; 
  __asm__ volatile("mftb %0" : "=r" (lsw)); 
  __asm__ volatile("mftbu %0" : "=r" (msw));
  return lsw + (((gm_tick_t)msw) << 32);
#elif GM_CPU_mips
#include <time.h>
  timespec_t t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec*(gm_u64_t)1000000000+t.tv_nsec;
#else
#warning do not how to define tick on your arch, using slow gettimeofday
#include <sys/time.h>
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec*(gm_u64_t)1000000+t.tv_usec;
#endif
}


#endif


#endif
