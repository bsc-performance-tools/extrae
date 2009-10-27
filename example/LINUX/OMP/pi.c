#include <stdio.h>
#include <omp.h>
#include <math.h>

int main(int argc, char **argv)
{
	int i;
	int n = 1000000;
	double PI25DT = 3.141592653589793238462643;
	double pi, h, area, x, start, end;

	OMPtrace_init();

	start = omp_get_wtime();
	h = 1.0 / (double) n;
	area = 0.0;
	#pragma omp parallel for private(x) reduction(+:area)
	for (i = 1; i <= n; i++)
	{
		x = h * ((double)i - 0.5);
		area += (4.0 / (1.0 + x*x));
	}
	pi = h * area;
	end = omp_get_wtime();

	printf("pi is approximately %.16f, Error is %.16f\n",pi,fabs(pi - PI25DT));
	printf("Ran in %0.5f seconds with %d threads\n",end-start,omp_get_max_threads());

	OMPtrace_fini();
}
