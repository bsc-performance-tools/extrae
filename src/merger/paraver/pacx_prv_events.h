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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef PACX_PRV_EVENTS_H
#define PACX_PRV_EVENTS_H

#include <config.h>

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "common.h"

/*************************************************************************
 * S'utilitza el format nou dels .prv, que genera diferents tipus i estan
 * definits en un fitxer de constants.
 *************************************************************************/

void Enable_PACX_Soft_Counter (unsigned int EvType);

/* S'afegeix el fitxer on hi ha totes les constants */
#include "PACX_EventEncoding.h"

#define PACXTYPE_FLAG_COLOR 9

#define NUM_PACX_BLOCK_GROUPS  8     /* Dels 12, de moment nomes 8 son diferents */

#define NUM_PACX_PRV_ELEMENTS 150    /* 127 */

#if 0
extern struct t_event_mpit2prv event_mpit2prv[];
extern struct t_prv_type_info prv_block_groups[];
#endif

void SoftCountersEvent_WriteEnabled_PACX_Operations (FILE *fd);
void MPITEvent_WriteEnabled_PACX_Operations (FILE * fd);
void Enable_PACX_Operation (int tmpit);

#if defined(PARALLEL_MERGE)
void Share_PACX_Softcounter_Operations (void);
void Share_PACX_Operations (void);
#endif

void Translate_PACX_MPIT2PRV (int typempit, UINT64 valuempit, int *typeprv, UINT64 *valueprv);

#if 0
/* MACRO per obtenir facilment un tipus de block (i=[0..NUM_BLOCK_GROUPS-1])*/
#define PRV_BLOCK_TYPE(i)  prv_block_groups[i].type
/* MACRO per obtenir facilment una etiqueta de block (i=[0..NUM_BLOCK_GROUPS-1])*/
#define PRV_BLOCK_LABEL(i) prv_block_groups[i].label
/* MACRO per obtenir facilment el color d'un block (i=[0..NUM_BLOCK_GROUPS-1])*/
#define PRV_BLOCK_COLOR(i) prv_block_groups[i].flag_color
#endif


#endif
