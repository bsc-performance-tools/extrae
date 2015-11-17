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

#include "heat.h"

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )
#define NB 8
/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
	double diff, sum=0.0;
	int nbx, bx, nby, by;
  
	nbx = 1;
	bx = sizex/nbx;
	nby = NB;
	by = sizey/nby;
	for (int ii=0; ii<nbx; ii++)
		for (int jj=0; jj<nby; jj++)
		{
			#pragma omp task  shared (sum) label (nested_comp)
			{
				double local_sum = 0.0;
 				for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
					for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++)
					{
						utmp[i*sizey+j]= 0.25 * (u[ i*sizey     + (j-1) ]+  // left
						  u[ i*sizey     + (j+1) ]+  // right
						  u[ (i-1)*sizey + j     ]+  // top
						  u[ (i+1)*sizey + j     ]); // bottom
						diff = utmp[i*sizey+j] - u[i*sizey + j];
						local_sum += diff * diff; 
					}
					sum += local_sum;
			}
		}

	#pragma omp taskwait
	printf ("Partial residual %lf\n", sum); 
	return sum;
}

/*
 * Blocked Red-Black solver: one iteration step
 */
double relax_redblack (double *u, unsigned sizex, unsigned sizey)
{
	double unew, diff, sum=0.0;
	int nbx, bx, nby, by;
	int lsw;

	nbx = NB;
	bx = sizex/nbx;
	nby = NB;
	by = sizey/nby;
	// Computing "Red" blocks
	for (int ii=0; ii<nbx; ii++)
	{
		lsw = ii%2;
		for (int jj=lsw; jj<nby; jj=jj+2) 
	            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
        	        for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++)
			{
	        	    unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
		            diff = unew - u[i*sizey+ j];
		            sum += diff * diff; 
	        	    u[i*sizey+j]=unew;
	     	  	 }
	}

	// Computing "Black" blocks
	for (int ii=0; ii<nbx; ii++)
	{
		lsw = (ii+1)%2;
		for (int jj=lsw; jj<nby; jj=jj+2) 
			for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
				for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++)
				{
					unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
					  u[ i*sizey	+ (j+1) ]+  // right
					  u[ (i-1)*sizey	+ j     ]+  // top
					  u[ (i+1)*sizey	+ j     ]); // bottom
					diff = unew - u[i*sizey+ j];
					sum += diff * diff; 
					u[i*sizey+j]=unew;
				}
	}

	return sum;
}

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
	double unew, diff, sum=0.0;
	int nbx, bx, nby, by;

	nbx = NB;
	bx = sizex/nbx;
	nby = NB;
	by = sizey/nby;
	for (int ii=0; ii<nbx; ii++)
		for (int jj=0; jj<nby; jj++) 
			for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
				for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++)
				{
					unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
					  u[ i*sizey	+ (j+1) ]+  // right
					  u[ (i-1)*sizey	+ j     ]+  // top
					  u[ (i+1)*sizey	+ j     ]); // bottom
					diff = unew - u[i*sizey+ j];
					sum += diff * diff; 
					u[i*sizey+j]=unew;
				}

	return sum;
}

