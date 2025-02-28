#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000

void
vec_mult()
{
	time_t t;
	int p[N], v1[N], v2[N], i = 0;

	srand((unsigned) time(&t));

	for (i = 0; i < N; i++)
	{
		p[i] = rand() % 1000;
		v1[i] = rand() % 1000;
		v2[i] = rand() % 1000;
	}

	#pragma omp target
	{
		#pragma omp parallel for
		for (i = 0; i < N; i++)
		{
			p[i] = v1[i] * v2[i];
		}
	}

	fprintf(stdout, "p = [");
	for (i = 0; i < N-1; i++)
	{
		fprintf(stdout, "%d, ", p[i]);
	}
	fprintf(stdout, "%d]\n", p[N-1]);
}

int
foo()
{
	int myvar = 1;
	int myarray[1000];

	#pragma omp target device(0) map(tofrom: myvar)
	{
		myvar *= 10;
	}

	#pragma omp target device(0) map(to: myvar) map(from: myarray)
	{
		myarray[42] = myvar;
	}

	return myarray[42];
}

int
main()
{
	vec_mult();

	return foo();
}
