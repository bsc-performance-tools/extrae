/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                   Extrae                                  *
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

#include "common.h"

#if defined(OS_AIX) || defined (OS_SOLARIS)

unsigned bswap32(unsigned value)
{
  unsigned newValue;
  char* pnewValue = (char*) &newValue;
  char* poldValue = (char*) &value;
 
  pnewValue[0] = poldValue[3];
  pnewValue[1] = poldValue[2];
  pnewValue[2] = poldValue[1];
  pnewValue[3] = poldValue[0];
 
  return newValue;
}

unsigned long long bswap64 (unsigned long long value)
{
  unsigned long long newValue;
  char* pnewValue = (char*) &newValue;
  char* poldValue = (char*) &value;
 
  pnewValue[0] = poldValue[7];
  pnewValue[1] = poldValue[6];
  pnewValue[2] = poldValue[5];
  pnewValue[3] = poldValue[4];
  pnewValue[4] = poldValue[3];
  pnewValue[5] = poldValue[2];
  pnewValue[6] = poldValue[1];
  pnewValue[7] = poldValue[0];
 
  return newValue;
}

#endif
