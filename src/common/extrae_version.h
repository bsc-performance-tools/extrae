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

#ifndef EXTRAE_VERSION_H_INCLUDED
#define EXTRAE_VERSION_H_INCLUDED

#define EXTRAE_VERSION_NUMBER(maj,min,rev) (((maj)<<16) | ((min)<<8) | (rev))
#define EXTRAE_VERSION_MAJOR(x)            (((x)>>16) & 0xff)
#define EXTRAE_VERSION_MINOR(x)            (((x)>>8) & 0xff)
#define EXTRAE_VERSION_REVISION(x)         ((x) & 0xff)

#define EXTRAE_VERSION                     EXTRAE_VERSION_NUMBER(3,4,3)

/* These macros can be used as:

#if EXTRAE_VERSION_MAJOR(EXTRAE_VERSION) == 2
 # do something specific for Extrae 2.x.y
 #if EXTRAE_VERSION_MINOR(EXTRAE_VERSION) == 4
  # do something specfic for Extrae 2.4.y
 #endif
#endif

*/

#endif /* EXTRAE_VERSION_H_INCLUDED */
