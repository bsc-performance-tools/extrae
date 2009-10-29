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
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef _TYPES_H_
#define _TYPES_H_

#include "config.h"

#ifdef HAVE_STDINT_H
   #include <stdint.h>
#endif
#ifdef HAVE_INTTYPES_H
   #include <inttypes.h>
#endif

#if defined(HAVE_INT64_T) && defined(HAVE_UINT64_T)
   /* If system has uint64_t/int64_t use them first */ 
   typedef int64_t  INT64;
   typedef uint64_t UINT64;
#else
   /* If system does not have UINT64/INT64, check which type can be used instead */
   #if defined(HAVE_LONG_LONG) && (SIZEOF_LONG_LONG == 8)
      /* long long occupies 8 bytes!, use it as UINT64/INT64 */
      typedef long long INT64;
      typedef unsigned long long UINT64;
   #elif defined(HAVE_LONG) && (SIZEOF_LONG == 8)
      /* long occupies 8 bytes!, use it as UINT64/INT64 */
      typedef long INT64;
      typedef unsigned long UINT64;
   #else
      #error "No 64-bit data type found"
   #endif
#endif

#if defined(HAVE_INT32_T) && defined(HAVE_UINT32_T)
   /* If system has uint32/int32 use them first */ 
   typedef int32_t  INT32;
   typedef uint32_t UINT32;
#else
   /* If system does not have UINT32/INT32, check which type can be used instead */
   #if defined(HAVE_INT) && (SIZEOF_INT == 4) 
      /* int occupies 4 bytes!, use it as UINT32/INT32 */
      typedef int INT32;
      typedef unsigned int UINT32;
   #else
      #error "No 32-bit data type found"
   #endif
#endif

#if defined(HAVE_INT16_T) && defined(HAVE_UINT16_T)
   /* If system has uint16/int16 use them first */ 
   typedef int16_t  INT16;
   typedef uint16_t UINT16;
#else
   /* If system does not have UINT16/INT16, check which type can be used instead */
   #if defined(HAVE_SHORT) && (SIZEOF_SHORT == 2) 
      /* int occupies 4 bytes!, use it as UINT16/INT16 */
      typedef int INT16;
      typedef unsigned int UINT16;
   #else
      #error "No 16-bit data type found"
   #endif
#endif

#if defined(HAVE_INT8_T) && defined(HAVE_UINT8_T)
   /* If system has uint8/int8 use them first */ 
   typedef int8_t  INT8;
   typedef uint8_t UINT8;
#else 
   /* If system does not have UINT8/INT8, check which type can be used instead */
   #if defined(HAVE_CHAR) && (SIZEOF_CHAR == 1) 
      /* char occupies 1 byte!, use it as UINT8/INT8 */
      typedef char INT8;
      typedef unsigned char UINT8;
   #else
      #error "No 8-bit data type found"
   #endif
#endif

typedef UINT64 STACK_ADDRESS;

#endif /* _TYPES_H_ */

