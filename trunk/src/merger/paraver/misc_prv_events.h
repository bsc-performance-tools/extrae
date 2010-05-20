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

#ifndef MISC_PRV_EVENTS_H
#define MISC_PRV_EVENTS_H

#if HAVE_STDIO_H
# include <stdio.h>
#endif

void MISCEvent_WriteEnabledOperations (FILE * fd, long long options);

#define MN_LINEAR_HOST_LABEL  "Linear host number"
#define MN_LINECARD_LABEL     "Linecard"
#define MN_HOST_LABEL         "Node inside linecard"

#define BG_TORUS_X            "BG X Coordinate in Torus"
#define BG_TORUS_Y            "BG Y Coordinate in Torus"
#define BG_TORUS_Z            "BG Z Coordinate in Torus"
#define BG_PROCESSOR_ID       "BG Processor ID"

#if defined(PARALLEL_MERGE)
void Share_MISC_Operations (void);
#endif

#endif
