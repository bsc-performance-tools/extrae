
#include <stdio.h>
#include <math.h>

double pi_kernel (int n, double h)
{
	double tmp = 0;
	double x;
	int i;

	for (i = 1; i <= n; i++)
	{
		x = h * ((double)i - 0.5);
		tmp += (4.0 / (1.0 + x*x));
	}

	return tmp;
}

int main(void)
{
	int n = 1000;
	double PI25DT = 3.141592653589793238462643;
	double pi, h, area;

	h = 1.0 / (double) n;
	area = pi_kernel (n, h);
	pi = h * area;

	printf("pi is approximately %.16f, Error is %.16f\n",pi,fabs(pi - PI25DT));

	return 0;
}
