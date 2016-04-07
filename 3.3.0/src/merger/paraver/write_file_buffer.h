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

#ifndef WRITE_BUFFER_H
#define WRITE_BUFFER_H

#include <config.h>

typedef struct 
{
	void *Buffer;
	off_t lastWrittenLocation;
	size_t sizeElement;
	int maxElements;
	int numElements;
	int FD;
	char *filename;
}
WriteFileBuffer_t;

WriteFileBuffer_t * WriteFileBuffer_new (int FD, char *filename, int maxElements, size_t sizeElement);
void WriteFileBuffer_delete (WriteFileBuffer_t *wfb);
void WriteFileBuffer_deleteall (void);
int WriteFileBuffer_getFD (WriteFileBuffer_t *wfb);
void WriteFileBuffer_flush (WriteFileBuffer_t *wfb);
off_t WriteFileBuffer_getPosition (WriteFileBuffer_t *wfb);
void WriteFileBuffer_write (WriteFileBuffer_t *wfb, const void* data);
void WriteFileBuffer_writeAt (WriteFileBuffer_t *wfb, const void* data, off_t position);
void WriteFileBuffer_removeLast (WriteFileBuffer_t *wfb);

#endif
