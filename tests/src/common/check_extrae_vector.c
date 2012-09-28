
#include <assert.h>
#include <extrae_vector.h>

#define UNREFERENCED_PARAMETER(x) ((x)=(x))

Extrae_Vector_t v;

int d = 4;
static int e = 5;

int main (int argc, char *argv[])
{
	int a = 1;
	int b = 2;
	int c = 3;

	UNREFERENCED_PARAMETER(argc);
	UNREFERENCED_PARAMETER(argv);
	
	Extrae_Vector_Init (&v);

	Extrae_Vector_Append (&v, &a);
	Extrae_Vector_Append (&v, &b);
	Extrae_Vector_Append (&v, &c);
	Extrae_Vector_Append (&v, &d);
	Extrae_Vector_Append (&v, &e);

	assert (Extrae_Vector_Count(&v) == 5);

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

	return 0;
}

