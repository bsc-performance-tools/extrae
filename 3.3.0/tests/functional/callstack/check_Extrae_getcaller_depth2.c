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

#include <stdint.h>
#include <stdlib.h>

extern uint64_t Extrae_get_caller (int);

void bar (void)
{
	uint64_t v;
	bar_begin:
	v = Extrae_get_caller (2);
	printf ("begin = %p v = %p end = %p\n", &&bar_begin, v, &&bar_end);
	if (v >= (uint64_t) &&bar_begin && v <= (uint64_t) &&bar_end)
	{
	}
	else
		exit (3);
	bar_end:
	return;
}

void foo (void)
{
	uint64_t v;
	foo_begin:
	v = Extrae_get_caller (2);
	printf ("begin = %p v = %p end = %p\n", &&foo_begin, v, &&foo_end);
	if (v >= (uint64_t) &&foo_begin && v <= (uint64_t) &&foo_end)
	{
		bar ();
	}
	else
		exit (2);
	foo_end:
	return;
}

int main (int argc, char *argv[])
{
	uint64_t v;
	argc = argc; argv = argv; /* Prevent unused warnings */
	main_begin:
	v = Extrae_get_caller (2);
	printf ("begin = %p v = %p end = %p\n", &&main_begin, v, &&main_end);
	if (v >= (uint64_t) &&main_begin && v <= (uint64_t) &&main_end)
	{
		foo ();
	}
	else
		exit (1);
	main_end:
	return 0;
}
