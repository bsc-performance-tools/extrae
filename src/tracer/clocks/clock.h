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

#ifndef CLOCK_H
#define CLOCK_H

enum
{
  REAL_CLOCK = 0,
  USER_CLOCK,
};

/* 
 * In Cell/64, the PPU is compiled in 64-bit, but the SPU is always compiled in 32-bit.
 * We need iotimer_t to be mapped into a 64-bit type both in the PPU and SPU, so we try to use unsigned long long first.
 */
#if SIZEOF_LONG_LONG == 8 
typedef unsigned long long iotimer_t;
#elif SIZEOF_LONG == 8
typedef unsigned long iotimer_t;
#endif

#define TIME (Clock_getTime())
#define CLOCK_INIT (Clock_Initialize())
#define CLOCK_INIT_THREAD (Clock_Initialize())

#if defined(__cplusplus)
extern "C" {
#endif
void Clock_setType (unsigned type);
unsigned Clock_getType (void);

iotimer_t Clock_getTime (void);
void Clock_Initialize (void);
void Clock_Initialize_thread (void);
#if defined(__cplusplus)
}
#endif

#endif
