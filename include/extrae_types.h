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

#ifndef EXTRAE_TYPES_INCLUDED
#define EXTRAE_TYPES_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

enum extrae_USER_COMMUNICATION_TYPES
{
	EXTRAE_USER_SEND = 0,
	EXTRAE_USER_RECV
};

enum extrae_USER_FUNCTION
{
	EXTRAE_USER_FUNCTION_NONE = -1,
	EXTRAE_USER_FUNCTION_LEAVE = 0,
	EXTRAE_USER_FUNCTION_ENTER
};

struct extrae_UserCommunication
{
	enum extrae_USER_COMMUNICATION_TYPES type;
	unsigned tag;
	unsigned size;
	unsigned partner;
	long long id;
};

struct extrae_CombinedEvents
{
	/* These are used as boolean values */
	int HardwareCounters;
	int Callers;
	int UserFunction;
	/* These are intended for N events */
	int nEvents;
	unsigned *Types;
	unsigned *Values;
	/* These are intended for user communication records */
	int nCommunications;
	struct extrae_UserCommunication *Communications;
};

#ifdef __cplusplus
}
#endif

#endif
