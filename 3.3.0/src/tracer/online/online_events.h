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

#ifndef __ONLINE_EVENTS_H__
#define __ONLINE_EVENTS_H__

/*
 * List of event types 
 */
enum
{
  ONLINE_STATE_EV = 666000,
  PERIODICITY_EV,
  DETAIL_LEVEL_EV,
  RAW_PERIODICITY_EV,
  RAW_BEST_ITERS_EV,
  HWC_MEASUREMENT
};

/* 
 * Values for event ONLINE_STATE_EV 
 */
enum
{
  ONLINE_RESUME_APP,
  ONLINE_PAUSE_APP
};

/* 
 * Values for event PERIODICITY_EV 
 */
enum
{
  NON_PERIODIC_ZONE = 0,
  REPRESENTATIVE_PERIOD
};

/*
 * Values for event DETAIL_LEVEL_EV
 */
enum
{
  NOT_TRACING = 0,
  PHASE_PROFILE,
  BURST_MODE,
  DETAIL_MODE
};
 

#endif /* __ONLINE_EVENTS_H__ */
