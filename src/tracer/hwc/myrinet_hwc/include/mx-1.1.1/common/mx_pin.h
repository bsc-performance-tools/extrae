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

/* modifications for MX kernel lib made by
 * Brice.Goglin@ens-lyon.org (LIP/INRIA/ENS-Lyon) */

#ifndef _mx_pin_h_
#define _mx_pin_h_

#include "mx_int.h"

typedef uintptr_t mx_pin_type_t;

/* define memory types that are passed to communication routines */
#define MX_PIN_UNDEFINED	(1UL << 0)

#ifdef MX_KERNEL
uintptr_t mx_klib_memory_context(void);

#define MX_PIN_KERNEL		(1UL << 1)
#define MX_PIN_USER		mx_klib_memory_context()
#define MX_PIN_PHYSICAL		(1UL << 2)

/* internal flags */
#define MX_PIN_STREAMING	(1UL << 4)
#define MX_PIN_CONSISTENT	(1UL << 5)

#define IS_USER_AS(x)	(x > (1UL << 5))
#define MX_AS_TO_PIN_FLAGS(x)  (IS_USER_AS(x)? 0 : (x))
#define MX_PIN_LIBMX_CTX MX_PIN_KERNEL

#else
/* user-level */
#define MX_PIN_LIBMX_CTX MX_PIN_UNDEFINED
#endif

#endif /* _mx_pin_h */
