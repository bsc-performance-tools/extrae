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

#ifndef __SEMANTICS_H_INCLUDED__
#define __SEMANTICS_H_INCLUDED__

#include "file_set.h"
#include "events.h"

enum
{
	PRV_SEMANTICS,
	TRF_SEMANTICS
};

typedef int Ev_Handler_t(event_t *, unsigned long long, unsigned int, unsigned int, unsigned int, unsigned int, FileSet_t *);

typedef struct
{
	int event;
	Ev_Handler_t *handler;
} SingleEv_Handler_t;

typedef struct
{
	int range_min;
	int range_max;
	Ev_Handler_t *handler;
} RangeEv_Handler_t;

/* public: */
void Semantics_Initialize (int output_format);
Ev_Handler_t * Semantics_getEventHandler (int event);
int SkipHandler (event_t *, unsigned long long, unsigned int, unsigned int, unsigned int, unsigned int, FileSet_t *);

#endif /* __SEMANTICS_H_INCLUDED__ */
