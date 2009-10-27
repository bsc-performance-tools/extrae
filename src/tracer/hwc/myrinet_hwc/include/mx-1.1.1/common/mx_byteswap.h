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

#ifndef MX_BYTESWAP_H
#define MX_BYTESWAP_H

#ifndef _mx_int_h_
#include "mx_int.h"
#endif

#ifndef MX_KERNEL
#if MX_OS_MACOSX
#include <machine/endian.h>
#elif MX_OS_LINUX || MX_OS_FREEBSD || MX_OS_UDRV
#include <netinet/in.h>
#elif MX_OS_SOLARIS
#include <sys/types.h>
#include <netinet/in.h>
#include <inttypes.h>
#elif MX_OS_WINNT
#include <winsock2.h>
#else
#error
#endif
#endif

#define mx_constant_swab32(x) \
(uint32_t)((((uint32_t)(x) >> 24) &   0xff) | \
           (((uint32_t)(x) >>  8) & 0xff00) | \
           (((uint32_t)(x) & 0xff00) <<  8) | \
           (((uint32_t)(x) &   0xff) << 24))

#define mx_constant_swab16(x) \
(uint16_t)((((uint16_t)(x) >> 8) & 0xff) | \
           (((uint16_t)(x) & 0xff) << 8))

#if MX_CPU_BIGENDIAN
#define mx_constant_htonl(x) (x)
#define mx_constant_ntohl(x) (x)
#define mx_constant_htons(x) (x)
#define mx_constant_ntohs(x) (x)
#else
#define mx_constant_htonl(x) mx_constant_swab32(x)
#define mx_constant_ntohl(x) mx_constant_swab32(x)
#define mx_constant_htons(x) mx_constant_swab16(x)
#define mx_constant_ntohs(x) mx_constant_swab16(x)
#endif

#if MX_CPU_BIGENDIAN
#define mx_htonll(x) (x)
#define mx_ntohll(x) (x)
#else
#define mx_htonll(x) \
(((x >> 56) &       0xff) + \
 ((x >> 40) &     0xff00) + \
 ((x >> 24) &   0xff0000) + \
 ((x >> 8)  & 0xff000000) + \
 ((x & 0xff000000) <<  8) + \
 ((x &   0xff0000) << 24) + \
 ((x &     0xff00) << 40) + \
 ((x &       0xff) << 56))
#define mx_ntohll(x) mx_htonll(x)
#endif

#endif
