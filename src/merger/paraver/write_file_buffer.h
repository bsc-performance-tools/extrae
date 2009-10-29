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
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef WRITE_BUFFER_H
#define WRITE_BUFFER_H

#include <config.h>

typedef struct 
{
	void *Buffer;
	off_t lastWrittenLocation;
	size_t sizeElement;
	int FD;
	int maxElements;
	int numElements;
}
WriteFileBuffer_t;

WriteFileBuffer_t * WriteFileBuffer_new (char *filename, int maxElements, size_t sizeElement);
void WriteFileBuffer_delete (WriteFileBuffer_t *wfb);
int WriteFileBuffer_getFD (WriteFileBuffer_t *wfb);
void WriteFileBuffer_flush (WriteFileBuffer_t *wfb);
off_t WriteFileBuffer_getPosition (WriteFileBuffer_t *wfb);
void WriteFileBuffer_write (WriteFileBuffer_t *wfb, const void* data);
void WriteFileBuffer_writeAt (WriteFileBuffer_t *wfb, const void* data, off_t position);
void WriteFileBuffer_removeLast (WriteFileBuffer_t *wfb);

#endif
