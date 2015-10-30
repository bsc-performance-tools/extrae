#include <stdio.h>

extern void fA(void);
extern void fB(void);
extern void Extrae_user_function (unsigned);

int main (int argc, char *argv[])
{
	Extrae_user_function(1);
	fA();
	fB();
	Extrae_user_function(0);
	return 0;
}
