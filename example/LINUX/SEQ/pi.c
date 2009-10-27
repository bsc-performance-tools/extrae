#include <stdio.h>
#include <math.h>

int main(int argc, char **argv)
{
	int i;
	int n = 1000000;
	double PI25DT = 3.141592653589793238462643;
	double pi, h, area, x;

	SEQtrace_init();

	h = 1.0 / (double) n;
	area = 0.0;

	SEQtrace_event (1000, 1);
	for (i = 1; i <= n; i++)
	{
		x = h * ((double)i - 0.5);
		area += (4.0 / (1.0 + x*x));
	}
	SEQtrace_event (1000, 0);
	pi = h * area;

	printf("pi is approximately %.16f, Error is %.16f\n",pi,fabs(pi - PI25DT));

	SEQtrace_fini();
}
