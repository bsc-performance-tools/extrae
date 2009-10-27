/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/parallel/mpi-tags.h,v $
 | 
 | @last_commit: $Date: 2008/01/21 09:51:28 $
 | @version:     $Revision: 1.2 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef MPI_TAGS_H_INCLUDED
#define MPI_TAGS_H_INCLUDED

#define REMAINING_TAG                  1000 /* Ask how many remaining events are on the merged intermediate file */

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

#endif

