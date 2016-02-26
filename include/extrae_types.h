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

#ifndef EXTRAE_TYPES_INCLUDED
#define EXTRAE_TYPES_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

enum extrae_INIT_TYPE
{
	EXTRAE_NOT_INITIALIZED = 0,
	EXTRAE_INITIALIZED_EXTRAE_INIT,
	EXTRAE_INITIALIZED_MPI_INIT,
	EXTRAE_INITIALIZED_SHMEM_INIT
};

typedef enum extrae_INIT_TYPE extrae_init_type_t;

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

typedef enum extrae_USER_FUNCTION  extrae_user_function_t;
typedef enum extrae_USER_COMMUNICATION_TYPES  extrae_user_communication_types_t;

typedef unsigned extrae_comm_tag_t;
typedef unsigned extrae_comm_partner_t;
typedef long long extrae_comm_id_t;
typedef unsigned extrae_type_t;
typedef unsigned long long extrae_value_t;

#define EXTRAE_COMM_PARTNER_MYSELF ((extrae_comm_partner_t) 0xFFFFFFFF)

struct extrae_UserCommunication
{
	extrae_user_communication_types_t type;
	extrae_comm_tag_t tag;
	unsigned size;
	extrae_comm_partner_t partner;
	extrae_comm_id_t id;
};

typedef struct extrae_UserCommunication  extrae_user_communication_t;

struct extrae_CombinedEvents
{
	/* These are used as boolean values */
	int HardwareCounters;
	int Callers;
	int UserFunction;
	/* These are intended for N events */
	unsigned nEvents;
	extrae_type_t  *Types;
	extrae_value_t *Values;
	/* These are intended for user communication records */
	unsigned nCommunications;
	extrae_user_communication_t *Communications;
};

typedef struct extrae_CombinedEvents  extrae_combined_events_t;

#ifdef __cplusplus
}
#endif

#endif
