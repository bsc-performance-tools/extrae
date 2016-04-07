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

#ifndef MPI_TAGS_H_INCLUDED
#define MPI_TAGS_H_INCLUDED

#define REMAINING_TAG                  1000 /* Ask how many remaining events are on the merged intermediate file */

#define NUMBER_SYNC_TIMES_TAG          1100 /* # of sync times to be sent recved */
#define SYNC_TIMES_TAG                 1101 /* sync times to be sent recved */
#define START_TIMES_TAG                1102 /* sync times to be sent recved */

#define ASK_MERGE_REMOTE_BLOCK_TAG     2000 /* Ask for the next remote merged block of events */
#define HOWMANY_MERGE_REMOTE_BLOCK_TAG 2001 /* Answer: how many events will be send */
#define BUFFER_MERGE_REMOTE_BLOCK_TAG  2002 /* Answer: actual buffer */

#define HOWMANY_FOREIGN_RECVS_TAG      3000 /* Tells how many foreign receives have been gathered */
#define BUFFER_FOREIGN_RECVS_TAG       3001 /* Actual buffer of foreign receives */

#define NUMBER_OF_HWC_SETS_TAG         4000 /* How many eventsets will be send */
#define HWC_SETS_READY                 4001 /* Let slaves send counters */
#define HWC_SETS_TAG                   4002 /* Which HWCs */
#define HWC_SETS_ENABLED_TAG           4003 /* Which HWCs are enabled */

#define DIMEMAS_CHUNK_FILE_SIZE_TAG    5000 /* Size of the partial translated TRF */
#define DIMEMAS_CHUNK_DATA_TAG         5001 /* Data of the partial translated TRF */

#define ADDRESSCOLLECTOR_ASK_TAG       6000 /* Ask for address collector info */
#define ADDRESSCOLLECTOR_NUM_TAG       6001 /* Number of addresses collected */
#define ADDRESSCOLLECTOR_ADDRESSES_TAG 6002 /* Addresses collected */
#define ADDRESSCOLLECTOR_TYPES_TAG     6003 /* Types collected */
#define ADDRESSCOLLECTOR_PTASKS_TAG    6004 /* PTasks involved in the address */
#define ADDRESSCOLLECTOR_TASKS_TAG     6005 /* Tasks involved in the address */

#endif

