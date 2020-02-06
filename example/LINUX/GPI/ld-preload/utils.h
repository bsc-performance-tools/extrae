#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#define ITERATIONS 1000

typedef unsigned long mcycles_t;

static inline mcycles_t
get_mcycles (void)
{
  unsigned low, high;
  unsigned long long val;

  __asm__ volatile ("rdtsc":"=a" (low), "=d" (high));
  val = high;
  val = (val << 32) | low;
  return val;
}

static int
mcycles_compare (const void *aptr, const void *bptr)
{
  const mcycles_t *a = (mcycles_t *) aptr;
  const mcycles_t *b = (mcycles_t *) bptr;
  if (*a < *b)
    return -1;
  if (*a > *b)
    return 1;
  return 0;
}

mcycles_t stamp[ITERATIONS], stamp2[ITERATIONS], delta[ITERATIONS];

#endif //_UTILS_H_
