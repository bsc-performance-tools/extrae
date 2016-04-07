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

/*
 * misc.c
 *
 * Helper functions for
 * - printing test parameters 
 * - rading parameters file
 * - initialization
 * - finalization,
 * - writing out a picture
 * - timing execution time
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#include "heat.h"

/*
 * Initialize the iterative solver
 * - allocate memory for matrices
 * - set boundary conditions according to configuration
 */
int initialize( algoparam_t *param )
{
	int i, j;
	double dist;

	// total number of points (including border)
	const int np = param->resolution + 2;
  
	//
	// allocate memory
	//
	(param->u)     = (double*)calloc( sizeof(double),np*np );
	(param->uhelp) = (double*)calloc( sizeof(double),np*np );
	(param->uvis)  = (double*)calloc( sizeof(double),
	  (param->visres+2) * (param->visres+2) );
  
	if( !(param->u) || !(param->uhelp) || !(param->uvis) )
	{
		fprintf(stderr, "Error: Cannot allocate memory\n");
		return 0;
	}

	for( i=0; i<param->numsrcs; i++ )
	{
		/* top row */
		for( j=0; j<np; j++ )
		{
			dist = sqrt( pow((double)j/(double)(np-1) - 
			  param->heatsrcs[i].posx, 2)+
			  pow(param->heatsrcs[i].posy, 2));
		  
			if( dist <= param->heatsrcs[i].range )
			{
				(param->u)[j] +=
				  (param->heatsrcs[i].range-dist) /
				  param->heatsrcs[i].range *
				  param->heatsrcs[i].temp;
			}
		}
	      
		/* bottom row */
		for( j=0; j<np; j++ )
		{
			dist = sqrt( pow((double)j/(double)(np-1) - 
			  param->heatsrcs[i].posx, 2)+
			  pow(1-param->heatsrcs[i].posy, 2));
		  
			if( dist <= param->heatsrcs[i].range )
			{
				(param->u)[(np-1)*np+j]+=
				  (param->heatsrcs[i].range-dist) / 
				  param->heatsrcs[i].range * 
				  param->heatsrcs[i].temp;
			}
		}
	      
		/* leftmost column */
		for( j=1; j<np-1; j++ )
		{
			dist = sqrt( pow(param->heatsrcs[i].posx, 2)+
			  pow((double)j/(double)(np-1) - 
			  param->heatsrcs[i].posy, 2)); 
		  
			if( dist <= param->heatsrcs[i].range )
			{
				(param->u)[ j*np ]+=
				  (param->heatsrcs[i].range-dist) / 
				  param->heatsrcs[i].range *
				  param->heatsrcs[i].temp;
			}
		}
	      
		/* rightmost column */
		for( j=1; j<np-1; j++ )
		{
			dist = sqrt( pow(1-param->heatsrcs[i].posx, 2)+
			  pow((double)j/(double)(np-1) - 
			  param->heatsrcs[i].posy, 2)); 
		  
			if( dist <= param->heatsrcs[i].range )
			{
				(param->u)[ j*np+(np-1) ]+=
				  (param->heatsrcs[i].range-dist) /
				  param->heatsrcs[i].range *
				  param->heatsrcs[i].temp;
			}
		}
	}

	// Copy u into uhelp
	double *putmp, *pu;
	pu = param->u;
	putmp = param->uhelp;
	for( int j=0; j<np; j++ )
		for( int i=0; i<np; i++ )
			*putmp++ = *pu++;

    return 1;
}

/*
 * free used memory
 */
int finalize( algoparam_t *param )
{
	if( param->u )
	{
		free(param->u);
		param->u = 0;
	}

	if( param->uhelp )
	{
		free(param->uhelp);
		param->uhelp = 0;
	}

	if( param->uvis )
	{
		free(param->uvis);
		param->uvis = 0;
	}

	return 1;
}


/*
 * write the given temperature u matrix to rgb values
 * and write the resulting image to file f
 */
void write_image( FILE * f, double *u,
		  unsigned sizex, unsigned sizey ) 
{
	// RGB table
	unsigned char r[1024], g[1024], b[1024];
	int i, j, k;
	double min, max;

	j=1023;

	// prepare RGB table
	for( i=0; i<256; i++ )
	{
		r[j]=255; g[j]=i; b[j]=0;
		j--;
	}
	for( i=0; i<256; i++ )
	{
		r[j]=255-i; g[j]=255; b[j]=0;
		j--;
	}
	for( i=0; i<256; i++ )
	{
		r[j]=0; g[j]=255; b[j]=i;
		j--;
	}
	for( i=0; i<256; i++ )
	{
		r[j]=0; g[j]=255-i; b[j]=255;
		j--;
	}

	min=DBL_MAX;
	max=-DBL_MAX;

	// find minimum and maximum 
	for( i=0; i<sizey; i++ )
	{
		for( j=0; j<sizex; j++ )
		{
		    if( u[i*sizex+j]>max )
				max=u[i*sizex+j];
		    if( u[i*sizex+j]<min )
				min=u[i*sizex+j];
		}
	}
  

	fprintf(f, "P3\n");
	fprintf(f, "%u %u\n", sizex, sizey);
	fprintf(f, "%u\n", 255);

	for( i=0; i<sizey; i++ )
	{
		for( j=0; j<sizex; j++ )
		{
		    k=(int)(1024.0*(u[i*sizex+j]-min)/(max-min));
		    fprintf(f, "%d %d %d  ", r[k], g[k], b[k]);
		}
		fprintf(f, "\n");
	}
}


int coarsen( double *uold, unsigned oldx, unsigned oldy ,
	     double *unew, unsigned newx, unsigned newy )
{
	int i, j;
	int stepx;
	int stepy;
	int stopx = newx;
	int stopy = newy;

	if (oldx>newx)
		stepx=oldx/newx;
	else
	{
		stepx=1;
		stopx=oldx;
	}

	if (oldy>newy)
		stepy=oldy/newy;
	else
	{
		stepy=1;
		stopy=oldy;
	}

	// NOTE: this only takes the top-left corner,
	// and doesnt' do any real coarsening 
	for( i=0; i<stopy-1; i++ )
		for( j=0; j<stopx-1; j++ )
		    unew[i*newx+j]=uold[i*oldx*stepy+j*stepx];

	return 1;
}

#define BUFSIZE 100
int read_input( FILE *infile, algoparam_t *param )
{
	int i, n;
	char buf[BUFSIZE];

	fgets(buf, BUFSIZE, infile);
	n = sscanf( buf, "%u", &(param->maxiter) );
	if( n!=1 )
		return 0;

	fgets(buf, BUFSIZE, infile);
	n = sscanf( buf, "%u", &(param->resolution) );
	if( n!=1 )
		return 0;

	param->visres = param->resolution;

	fgets(buf, BUFSIZE, infile);
	n = sscanf(buf, "%d", &(param->algorithm) );
	if( n!=1 )
		return 0;

	fgets(buf, BUFSIZE, infile);
	n = sscanf(buf, "%u", &(param->numsrcs) );
	if( n!=1 )
		return 0;

	(param->heatsrcs) = 
		(heatsrc_t*) malloc( sizeof(heatsrc_t) * (param->numsrcs) );
  
	for( i=0; i<param->numsrcs; i++ )
	{
		fgets(buf, BUFSIZE, infile);
		n = sscanf( buf, "%f %f %f %f",
		  &(param->heatsrcs[i].posx),
		  &(param->heatsrcs[i].posy),
		  &(param->heatsrcs[i].range),
		  &(param->heatsrcs[i].temp) );

		if( n!=4 )
			return 0;
	}

	return 1;
}


void print_params( algoparam_t *param )
{
	int i;

	fprintf(stdout, "Iterations        : %u\n", param->maxiter);
	fprintf(stdout, "Resolution        : %u\n", param->resolution);
	fprintf(stdout, "Algorithm         : %d (%s)\n",
	  param->algorithm,
	  (param->algorithm == 0) ? "Jacobi":(param->algorithm ==1) ? "Red-Black":"Gauss-Seidel");
	fprintf(stdout, "Num. Heat sources : %u\n", param->numsrcs);

	for( i=0; i<param->numsrcs; i++ )
	{
		fprintf(stdout, "  %2d: (%2.2f, %2.2f) %2.2f %2.2f \n",
	     i+1,
	     param->heatsrcs[i].posx,
	     param->heatsrcs[i].posy,
	     param->heatsrcs[i].range,
	     param->heatsrcs[i].temp );
	}
}


//
// timing.c 
// 
double wtime(void)
{
	struct timeval tv;
	gettimeofday(&tv, 0);

	return tv.tv_sec+1e-6*tv.tv_usec;
}

