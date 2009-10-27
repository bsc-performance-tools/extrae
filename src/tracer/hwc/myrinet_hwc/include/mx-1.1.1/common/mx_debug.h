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
 * Copyright 2003 - 2004 by Myricom, Inc.  All rights reserved.          *
 *************************************************************************/

#ifndef _mx_debug_h_
#define _mx_debug_h_

#include "mx_auto_config.h"

#ifndef MX_KERNEL
#define MX_PRINT(s)	mx_printf s
#define MX_WARN(s)	mx_printf s
#endif

#define MX_DEBUG_ALL    (uint32_t)-1;

#define MX_NOTE MX_PRINT
void
mx_assertion_failed (const char *assertion, int line, const char *file);

void mx__abort(void);

#define mx_always_assert(a) _mx_always_assert (a, #a)
#define mx_fixme_assert mx_always_assert
#define _mx_always_assert(a,txt) do {                   \
  if (!(a)) {                                           \
    mx_assertion_failed (txt, __LINE__, __FILE__);      \
  }                                                     \
} while (0)
#define mx_fatal(a) mx_assertion_failed(a, __LINE__, __FILE__)

#if MX_OS_WINNT
#define mx_printf_once mx_printf
#else
#define mx_printf_once(...) do { static int _deja_vu; \
       if (!_deja_vu++) { mx_printf( __VA_ARGS__ ) ; } \
     } while (0)
#endif

extern uint32_t        mx_debug_mask;

#if MX_DEBUG

#define MX_DEBUG_INC(mask, x)           \
  do {                                  \
    if ((mask) & mx_debug_mask)         \
      x++;                              \
   } while (0)

#define MX_DEBUG_PRINT(mask, s)         \
  do {                                  \
    if ((mask) & mx_debug_mask)         \
      MX_PRINT (s);                     \
  } while (0)

#define mx_assert(a) _mx_always_assert (a, #a)
#else
#define MX_DEBUG_INC(mask, x)
#define MX_DEBUG_PRINT(mask, s)
#define mx_assert(a) do {if (0) (void) (a);} while (0) /* allow syntax check */
#endif

#define MX_DEBUG_BOARD_INIT (1 << 0)
#define MX_TRACE_LANAI_DMA  (1 << 1)
#define MX_DEBUG_MALLOC     (1 << 2)
#define MX_DEBUG_OPENCLOSE  (1 << 3)
#define MX_DEBUG_SLEEP      (1 << 4)
#define MX_DEBUG_KVA_TO_PHYS (1 << 5)
#define MX_DEBUG_INTR       (1 << 6)
#define MX_DEBUG_MAPPER     (1 << 7)

#define MX_DEBUG_REQUESTS   (1 << 0)
#define MX_DEBUG_EVENTS     (1 << 1)
#define MX_DEBUG_OPEN_CLOSE (1 << 2)

/* definitions for debugging TCP library */
#define MX_DEBUG_TCP_LIB    (1 << 0)

#define MX_VAR_MAY_BE_UNUSED(v) ((void)(v))
#define MX_PARAMETER_MAY_BE_UNUSED(v) ((void)(v))

#endif  /* _mx_debug_h_ */
