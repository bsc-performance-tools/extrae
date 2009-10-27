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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/common/mpit_header.h,v $
 | 
 | @last_commit: $Date: 2009/04/29 15:44:53 $
 | @version:     $Revision: 1.2 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __MPIT_HEADER_H__
#define __MPIT_HEADER_H__

typedef struct 
{
	int Signature;
	int Version;
	int CircularBuffer;
	int FirstTime;
	int SyncTime;
	int LastTime;
	int StartingTracingMode;
	int Endianness;
	int bits; /* 32 o 64 */
	int HWCs;
	int HWCsLabels;
	int ClockType; // REAL_CLOCK? PRV/DIM
	int HostInfo;
	int ConfigXML;
} MPIT_Header_t;

/*
 Write header
??? Si buffer circular emitir un 1er evento que sea una marca de buffer overflow para controlar si es un pedazo de traza intermedio... 
*/

#endif /* __MPIT_HEADER_H__ */
