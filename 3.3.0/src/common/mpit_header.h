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
