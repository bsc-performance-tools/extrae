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

#ifndef __MRNET_ROOT_H__
#define __MRNET_ROOT_H__

#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 255
#endif

#define TOPOLOGY_FILE_TEMPLATE "top.XXXXXX"

#include "BEMask.h"
#include "mrnet/MRNet.h"
using namespace MRN;

/* Prototypes */
#if defined(__cplusplus)
extern "C" {
#endif


int Start_MRNet(int argc, char ** argv);
Stream * Announce_Stream (Network *n, Stream *bcast_stream, BEMask *be_mask, int up_transfilter_id, int up_syncfilter_id);


#if defined(__cplusplus)
}
#endif

#endif /* __MRNET_ROOT_H__ */
