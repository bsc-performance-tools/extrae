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

#ifndef _mx_int_h_
#define _mx_int_h_

#include "mx_autodetect.h"

#ifdef MX_MCP
typedef signed char          int8_t;
typedef signed short        int16_t;
typedef signed int          int32_t;
typedef signed long long    int64_t;
typedef unsigned char       uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
#define MX_PRI_64 "ll"
#elif MX_OS_LINUX
#ifdef MX_KERNEL
#include <linux/types.h>
typedef long intptr_t;
typedef unsigned long uintptr_t;
#if BITS_PER_LONG == 32 || MX_CPU_x86_64
#define MX_PRI_64 "ll"
#define UINT64_C(value) value##ULL
#elif BITS_PER_LONG == 64
#define MX_PRI_64 "l"
#define UINT64_C(value) value##UL
#else
#error 32 or 64?
#endif
#else
#include <inttypes.h>
#include <stdint.h>
#endif
#elif MX_OS_MACOSX || MX_OS_FREEBSD 
#ifdef MX_KERNEL
#include <sys/types.h>
#if MX_OS_MACOSX
#include <machine/limits.h>
#include <stdint.h>
#else
#include <sys/limits.h>
#endif
#if LONG_BIT == 32
#define MX_PRI_64 "ll"
#elif LONG_BIT == 64
#define MX_PRI_64 "l"
#endif
#else
/* MACOSX inttypes.h doesn't define SCNx64 unless __STDC_LIBRARY_SUPPORTED__ */
#if MX_OS_MACOSX
#ifndef __STDC_LIBRARY_SUPPORTED__
#define __STDC_LIBRARY_SUPPORTED__
#endif
#endif
#include <inttypes.h>
#include <stdint.h>
#endif
#elif MX_OS_SOLARIS
#include <inttypes.h>
#elif MX_OS_AIX
#include <sys/types.h>
#elif MX_OS_WINNT
typedef signed __int8      int8_t;
typedef signed __int16    int16_t;
typedef signed __int32    int32_t;
typedef signed __int64    int64_t;
typedef unsigned __int8   uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#ifdef _M_IX86
typedef unsigned __int32 uintptr_t;
#else
typedef unsigned __int64 uintptr_t;
#endif
#define MX_PRI_64 "I64"
#define MX_SCN_64 MX_PRI_64
#define UINT64_C(value) value##ull
#elif MX_OS_UDRV
#include <inttypes.h>
#include <stdint.h>
#else
#error Your platform is unsupported
#endif


#if defined MX_PRI_64 && !defined PRIx64
/* we define MX_PRI_64 no definition already present */
#define PRIx64 MX_PRI_64 "x"
#define PRId64 MX_PRI_64 "d"
#define PRIu64 MX_PRI_64 "u"
#endif
#if defined MX_SCN_64 && !defined SCNx64
/* we define MX_PRI_64 no definition already present */
#define SCNx64 MX_SCN_64 "x"
#define SCNd64 MX_SCN_64 "d"
#define SCNu64 MX_SCN_64 "u"
#endif

#ifdef MX_KERNEL
#if MX_OS_LINUX
typedef unsigned long mx_uaddr_t;
#elif MX_OS_MACOSX
#if MX_DARWIN_XX >= 8
typedef user_addr_t mx_uaddr_t;
#else /* MX_DARWIN_XX */
typedef uintptr_t mx_uaddr_t;
#endif /* MX_DARWIN_XX */
#elif MX_OS_UDRV
typedef uint64_t mx_uaddr_t;
#else
typedef uintptr_t mx_uaddr_t;
#endif /* MX_OS_ */
#endif /* MX_KERNEL */

#endif /* _mx_int_h_ */
