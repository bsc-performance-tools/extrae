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

#include <unistd.h>
#include <pthread.h>

#include "extrae_user_events.h"

void * Task1(void *param)
{
	struct extrae_CombinedEvents events;
	struct extrae_UserCommunication comm;
	unsigned types[2] = { 123456, 123457 } ;
	unsigned values[2] = { 1, 2 };

	Extrae_init_UserCommunication (&comm);
	comm.type = EXTRAE_USER_RECV;
	comm.partner = 0;
	comm.tag = 1234;
	comm.size = 1024;
	comm.id = 0xdeadbeef;

	Extrae_init_CombinedEvents (&events);
	events.nCommunications = 1;
	events.Communications = &comm;
	events.nEvents = 2;
	events.Types = types;
	events.Values = values;

	Extrae_emit_CombinedEvents (&events);
}

void * Task0(void *param)
{
	struct extrae_CombinedEvents events;
	struct extrae_UserCommunication comm;
	unsigned types[2] = { 123456, 123457 } ;
	unsigned values[2] = { 1, 2 };

	Extrae_init_UserCommunication (&comm);
	comm.type = EXTRAE_USER_SEND;
	comm.partner = 0;
	comm.tag = 1234;
	comm.size = 1024;
	comm.id = 0xdeadbeef;

	Extrae_init_CombinedEvents (&events);
	events.nCommunications = 1;
	events.Communications = &comm;
	events.nEvents = 2;
	events.Types = types;
	events.Values = values;

	Extrae_emit_CombinedEvents (&events);
}

int main (int argc, char *argv[])
{
	pthread_t t[2];

	Extrae_init();
	pthread_create (&t[0], NULL, Task0, NULL);
	pthread_create (&t[1], NULL, Task1, NULL);
	pthread_join (t[0], NULL);
	pthread_join (t[1], NULL);
	Extrae_fini();

	return 0;
}

