#include <stdio.h>

extern void Extrae_user_function (unsigned);

void fA(void)
{
	Extrae_user_function(1);
	printf ("fA\n");
	Extrae_user_function(0);
}
