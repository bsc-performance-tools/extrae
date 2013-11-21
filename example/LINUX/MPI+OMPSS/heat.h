/*
 * heat.h
 *
 * Global definitions for the iterative solver
 */

#include <stdio.h>

// configuration

typedef struct
{
    float posx;
    float posy;
    float range;
    float temp;
}
heatsrc_t;

typedef struct
{
    unsigned maxiter;       // maximum number of iterations
    unsigned resolution;    // spatial resolution
    int algorithm;          // 0=>Jacobi, 1=>Gauss

    unsigned visres;        // visualization resolution
  
    double *u, *uhelp;
    double *uvis;

    unsigned   numsrcs;     // number of heat sources
    heatsrc_t *heatsrcs;
}
algoparam_t;

// function declarations

// misc.c
int initialize( algoparam_t *param );
int finalize( algoparam_t *param );
void write_image( FILE * f, double *u,
		  unsigned sizex, unsigned sizey );
int coarsen(double *uold, unsigned oldx, unsigned oldy ,
	    double *unew, unsigned newx, unsigned newy );
int read_input( FILE *infile, algoparam_t *param );
void print_params( algoparam_t *param );
double wtime(void);

// solvers in solver.c
double relax_redblack( double *u, 
		  unsigned sizex, unsigned sizey );

double relax_gauss( double *u, 
		  unsigned sizex, unsigned sizey );

double relax_jacobi( double *u, double *utmp,
		   unsigned sizex, unsigned sizey ); 

