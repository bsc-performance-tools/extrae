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
 * Iterative solver for heat distribution
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "heat.h"

void usage( char *s )
{
	fprintf(stderr, "Usage: %s <input file> [result file]\n\n", s);
}

int main( int argc, char *argv[] )
{
	unsigned iter;
	FILE *infile, *resfile;
	char *resfilename;
	int myid, numprocs;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	if (myid == 0)
	{
		// algorithmic parameters
		algoparam_t param;
		int np;
	
		double runtime, flop;
		double residual=0.0;
	
		// check arguments
		if( argc < 2 )
		{
			usage( argv[0] );
			return 1;
		}
	
		// check input file
		if( !(infile=fopen(argv[1], "r"))  ) 
		{
			fprintf(stderr, "\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);
			usage(argv[0]);
			return 1;
		}
	
		// check result file
		resfilename= (argc>=3) ? argv[2]:"heat.ppm";
	
		if( !(resfile=fopen(resfilename, "w")) )
		{
			fprintf(stderr, "\nError: Cannot open \"%s\" for writing.\n\n", resfilename);
			usage(argv[0]);
			return 1;
		}
	
		// check input
		if( !read_input(infile, &param) )
		{
			fprintf(stderr, "\nError: Error parsing input file.\n\n");
			usage(argv[0]);
			return 1;
		}
		print_params(&param);
	
		// set the visualization resolution
		param.u     = 0;
		param.uhelp = 0;
		param.uvis  = 0;
		param.visres = param.resolution;
		if (!initialize(&param) )
		{
			fprintf(stderr, "Error in Solver initialization.\n\n");
			usage(argv[0]);
			return 1;
		}
	
		// full size (param.resolution are only the inner points)
		np = param.resolution + 2;
	
		// starting time
		runtime = wtime();
	
		// send to workers the necessary data to perform computation
		int first_row = 1;
		int last_row;
		for (int i=0; i<numprocs; i++)
		{
			last_row = first_row + param.resolution/numprocs - 1;
			last_row = (i < (param.resolution%numprocs)) ? last_row+1 : last_row;
			if (i>0)
			{
				MPI_Send(&param.maxiter, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&param.resolution, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&param.algorithm, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&first_row, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&last_row, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&param.u[(first_row-1)*np], (last_row-first_row+3)*np, 
				  MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
				MPI_Send(&param.uhelp[(first_row-1)*np], (last_row-first_row+3)*np, 
				  MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			}
			first_row = last_row + 1;
		} 
	
		first_row = 1;
		last_row = first_row + param.resolution/numprocs - 1;
		last_row = ((param.resolution%numprocs)>0) ? last_row+1 : last_row;
		int rows = last_row-first_row+1;
	
		iter = 0;
		while(1)
		{
			switch( param.algorithm )
			{
				case 0: // JACOBI
				{
					double *uu, *uhelp;
					uu = param.u; 
					uhelp = param.uhelp;
					#pragma omp task in (*uu) out (*uhelp) out (residual) label (compute)
					residual = relax_jacobi(uu, uhelp, rows+2, np);
					printf ("Residual in main %lf\n", residual); 
					#pragma omp task inout (*uhelp) label (comm) inout (residual)
					{
						MPI_Sendrecv(&uhelp[last_row*np], np, MPI_DOUBLE, 1, 0,
						  &uhelp[(last_row+1)*np], np, MPI_DOUBLE, 1, 0,
						  MPI_COMM_WORLD, &status);
					}
					// Copy uhelp into u
					#pragma omp task in (*uhelp) out (*uu) label (copy)
					for (int i=first_row-1; i<last_row+2; i++)
						for (int j=0; j<np; j++)
							uu[ i*np+j ] = uhelp[ i*np+j ];
				}
			    break;
				case 1: // RED-BLACK
				    residual = relax_redblack(param.u, np, np);
				break;
				case 2: // GAUSS
					residual = relax_gauss(param.u, np, np);
				break;
			}
			iter++;
	
			// solution good enough ?
			#pragma omp task inout (residual) label (allreduce)
			MPI_Allreduce(MPI_IN_PLACE, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
			if (iter %10 ==0)
			{
				#pragma omp taskwait

				if (residual < 0.00005) break;
	
				// max. iteration reached ? (no limit with maxiter=0)
				if (param.maxiter>0 && iter>=param.maxiter)
					break;
			}
		}
	
		#pragma omp taskwait
		// receive from workers the portions of matrix u computed
		first_row = 1;
		for (int i=0; i<numprocs; i++)
		{
			last_row = first_row + param.resolution/numprocs - 1;
			last_row = (i < (param.resolution%numprocs)) ? last_row+1 : last_row;
			if (i>0)
				MPI_Recv(&param.u[first_row*np], (last_row-first_row+1)*(np), 
				  MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
			first_row = last_row + 1;
		} 
	
		// Flop count after iter iterations
		flop = iter * 11.0 * param.resolution * param.resolution;
	
		// stopping time
		runtime = wtime() - runtime;
	
		fprintf(stdout, "Time: %04.3f ", runtime);
		fprintf(stdout, "(%3.3f GFlop => %6.2f MFlop/s)\n", 
		  flop/1000000000.0,
		  flop/runtime/1000000);
		fprintf(stdout, "Convergence to residual=%f: %d iterations\n", residual, iter);
	
		// for plot...
		coarsen( param.u, np, np,
		  param.uvis, param.visres+2, param.visres+2 );
	
		write_image( resfile, param.uvis,  
		  param.visres+2, 
		  param.visres+2 );
	
		finalize( &param );
	
		MPI_Finalize();
	}
	else
	{

		algoparam_t param;

		// receive information from master to perform computation locally
		int  rows, np;
		int first_row, last_row;
		int iter;
		double residual;

		MPI_Recv(&param.maxiter, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&param.resolution, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&param.algorithm, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&first_row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&last_row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
	
		rows = last_row-first_row+1;
		np = param.resolution + 2;
	
		// allocate memory for worker
		param.u = calloc( sizeof(double), (rows+2)*np );
		param.uhelp = calloc( sizeof(double), (rows+2)*np );
		if( (!param.u) || (!param.uhelp) )
		{
			fprintf(stderr, "Error: Cannot allocate memory\n");
			return 0;
		}
	    
		// fill initial values for matrix with values received from master
		MPI_Recv(param.u, (rows+2)*np, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(param.uhelp, (rows+2)*np, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
	
		iter = 0;
		while(1)
		{
			switch( param.algorithm )
			{
				case 0: // JACOBI
				{
					double *uu, *uhelp;
					uu = param.u;
					uhelp = param.uhelp;
					#pragma omp task in (*uu) out (*uhelp)  out (residual) label (compute)
					residual = relax_jacobi(uu, uhelp, rows+2, np);
					#pragma omp task inout (*uhelp) label (comm) inout (residual)
					{
						if (myid < (numprocs-1))
						{
							MPI_Sendrecv(&uhelp[rows*np], np, MPI_DOUBLE, myid+1, 0,
							  &uhelp[(rows+1)*np], np, MPI_DOUBLE, myid+1, 0,
							  MPI_COMM_WORLD, &status);
						}
						MPI_Sendrecv(&uhelp[np], np, MPI_DOUBLE, myid-1, 0,
						  &uhelp[0], np, MPI_DOUBLE, myid-1, 0,
						  MPI_COMM_WORLD, &status);
					}
					// Copy uhelp into u
					#pragma omp task in (*uhelp) out(*uu) label (copy)
					for (int i=0; i<rows+2; i++)
						for (int j=0; j<np; j++)
							uu[ i*np+j ] = uhelp[ i*np+j ];
				}
				break;
				case 1: // RED-BLACK
					residual = relax_redblack(param.u, np, np);
				break;
				case 2: // GAUSS
					residual = relax_gauss(param.u, np, np);
				break;
			}
	
			iter++;
	
			// solution good enough ?
			#pragma omp task inout (residual)  label (allreduce)
			MPI_Allreduce(MPI_IN_PLACE, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
			if (iter %10 == 0)
			{
				#pragma omp taskwait 
	
				if (residual < 0.00005) break;
	
				// max. iteration reached ? (no limit with maxiter=0)
				if (param.maxiter>0 && iter>=param.maxiter) break;
			}
		}
	
		#pragma omp taskwait
		
		// send values computed for matrix to the master
		MPI_Send(&param.u[np], rows*np, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	
		if( param.u )
			free(param.u);
		if( param.uhelp )
			free(param.uhelp);
	
		MPI_Finalize();
	}

	return 0;
}
