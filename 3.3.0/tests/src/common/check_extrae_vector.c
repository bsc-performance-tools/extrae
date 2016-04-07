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


#include <assert.h>
#include <extrae_vector.h>

#define UNREFERENCED_PARAMETER(x) ((x)=(x))

Extrae_Vector_t v;
Extrae_Vector_t v2;

#define N 64

static int vec[N];

int d = 4;
static int e = 5;

int main (int argc, char *argv[])
{
	int i;
	int a = 1;
	int b = 2;
	int c = 3;

	for (i = 0; i < N; i++)
		vec[i] = i;

	UNREFERENCED_PARAMETER(argc);
	UNREFERENCED_PARAMETER(argv);
	
	Extrae_Vector_Init (&v);
	Extrae_Vector_Init (&v2);

	Extrae_Vector_Append (&v, &a);
	Extrae_Vector_Append (&v, &b);
	Extrae_Vector_Append (&v, &c);
	Extrae_Vector_Append (&v, &d);
	Extrae_Vector_Append (&v, &e);

	assert (Extrae_Vector_Count(&v) == 5);
	assert (Extrae_Vector_Count(&v2) == 0);

	assert (Extrae_Vector_Get(&v,0) == &a);
	assert ((*(int*) Extrae_Vector_Get(&v,0)) == a);

	assert (Extrae_Vector_Get(&v,1) == &b);
	assert ((*(int*) Extrae_Vector_Get(&v,1)) == b);

	assert (Extrae_Vector_Get(&v,2) == &c);
	assert ((*(int*) Extrae_Vector_Get(&v,2)) == c);

	assert (Extrae_Vector_Get(&v,3) == &d);
	assert ((*(int*) Extrae_Vector_Get(&v,3)) == d);

	assert (Extrae_Vector_Get(&v,4) == &e);
	assert ((*(int*) Extrae_Vector_Get(&v,4)) == e);

	Extrae_Vector_Destroy (&v);
	assert (Extrae_Vector_Count(&v) == 0);

	for (i = 0; i < N; i++)
		Extrae_Vector_Append (&v2, &vec[i]);

	assert (Extrae_Vector_Count(&v2) == N);

	for (i = 0; i < N; i++)
	{
		assert (Extrae_Vector_Get(&v2,i) == &vec[i]);
		assert ((*(int*) Extrae_Vector_Get(&v2,i)) == vec[i]);
	}

	Extrae_Vector_Destroy (&v2);
	assert (Extrae_Vector_Count(&v2) == 0);

	return 0;
}

