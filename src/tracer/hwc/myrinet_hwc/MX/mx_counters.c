/*
 * This small MPI program tries to illustrate the use of the reading of the
 * mx counters within a small MPI app.
 *
 * COMPILE:
 *

 See makefile

 * ONLY WORKS WITH SAME VERSION AS COMPILED KERNEL DRIVER
 */

//#define DEBUG 1
//#define MY_MAIN 1
//#define MPITRACE
//#define MX_PRINT 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define MPITRACE

/* This program uses undocumented software interfaces that are not
   intended for use in code written outside Myricom.  Hence the
   following undocumented includes. */
//#include "mx_auto_config.h"
#include "myriexpress.h"
#include "mx__driver_interface.h"
#include "mx__fops.h"
#ifdef MX_MARENOSTRUM_API
#include "mx_dispersion.h"
#endif

#ifdef MY_MAIN
#include "mpi.h"
#endif

#include "mxmpi.h"
//#include "mpid.h"

#include "mx_counters.h"

extern struct mxmpi_var mxmpi;
extern int    MPID_MyWorldSize, MPID_MyWorldRank;  //mpid.h

//#define MY_MAX_NODE (3*1024)
#define MY_MAX_NODE (6655)
#ifdef MX_MARENOSTRUM_API
#define MY_MAX_ROUTES 8
#endif
//#define DO_CHECKS

uint32_t *counters = NULL;
uint32_t *counters_previous = NULL;

#ifdef MX_MARENOSTRUM_API
uint32_t **dispersion_counters = NULL;
/* [MY_MAX_NODE][MY_MAX_ROUTES]; */
uint32_t **dispersion_counters_previous = NULL;
/* [MY_MAX_NODE][MY_MAX_ROUTES]; */
#endif

char *names;
uint32_t count;
uint32_t allocated_count;
#ifdef MX_MARENOSTRUM_API
uint32_t dispersion_count;
uint32_t dispersion_allocated_count;
#endif


uint32_t board_id = 0;

static int counters_previous_defined = 0;
#ifdef MX_MARENOSTRUM_API
static int dispersion_counters_previous_defined = 0;
#endif

#define SWAP(x,y) { uint32_t *ptr; ptr = x; x = y; y = ptr; }

#define CHECK_GLOBALS() { if ( counters == NULL ) { MYRINET_start_counters(); } }
#define CHECK_DISP_GLOBALS() { if ( counters == NULL ) { MYRINET_start_counters(); } }

#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define MIN(x,y) (((x) < (y)) ? (x) : (y))

int MYRINET_start_counters () 
{
	mx_return_t ret;

	if (counters == NULL || counters_previous == NULL) 
	{
		/* COUNTERS COUNT */
		ret = mx_get_info(NULL, MX_COUNTERS_COUNT, &board_id, sizeof (board_id), &count, sizeof (count));
		if (ret != MX_SUCCESS) 
		{
			perror("get counters count failed");
		}
		if (count == 0) 
		{
			fprintf(stderr,"mpich_read_counters: count (of counters) == 0\n");
			mxmpi_abort(MX_FAILURE);
		}

		/* ALLOCATE_COUNTERS */
		// allocated_count = MAX(1024,count);
		allocated_count = count;
		counters = (uint32_t *)malloc (allocated_count * sizeof(uint32_t));
		counters_previous = (uint32_t *)malloc (allocated_count * sizeof(uint32_t));
		if (counters == NULL || counters_previous == NULL) 
		{
			fprintf(stderr,"counters: out of memory\n");
			mxmpi_abort(MX_FAILURE);
		}

		/* ALLOCATE NAMES */
		names = (char *)malloc (MX_MAX_STR_LEN * allocated_count);
		if (names == NULL) 
		{
			fprintf(stderr,"names: out of memory\n");
			mxmpi_abort(MX_FAILURE);
		}
             
		/* POPULATE NAMES */
		ret = mx_get_info(NULL, MX_COUNTERS_LABELS, &board_id, sizeof (board_id), names, MX_MAX_STR_LEN * allocated_count);
		if (ret != MX_SUCCESS) 
		{
			perror("get counters strings failed");
			mxmpi_abort(MX_FAILURE);
		}
	}

#ifdef MX_MARENOSTRUM_API
	{
		int i;
		uint32_t * counters_basse_offset;
		uint32_t * counters_basse_offset_previous;

		if ( MPID_MyWorldSize > 0) 
		{
			dispersion_allocated_count = MAX ( MY_MAX_NODE, MPID_MyWorldSize );
			/* Should be MIN, but just in case */
		}
		dispersion_counters = (uint32_t **)malloc ( (dispersion_allocated_count * sizeof(uint32_t *) ) /* pointers */ +
				(dispersion_allocated_count * MY_MAX_ROUTES * sizeof(uint32_t)) /* counters */ );
		dispersion_counters_previous = (uint32_t **)malloc ((dispersion_allocated_count * sizeof(uint32_t *) ) /* pointers */ +
				(dispersion_allocated_count * MY_MAX_ROUTES * sizeof(uint32_t)) /* counters */ );

		/*fprintf(stderr,"dispersion_counters: allocating for %d nodes with %d routes each\n",dispersion_allocated_count, MY_MAX_ROUTES);*/
		if (dispersion_counters == NULL || dispersion_counters_previous == NULL) 
		{
			fprintf(stderr,"dispersion_counters: out of memory\n");
			mxmpi_abort(MX_FAILURE);
		}

		memset (dispersion_counters, 0, (dispersion_allocated_count * sizeof(uint32_t *) ) /* pointers */ +
			(dispersion_allocated_count * MY_MAX_ROUTES * sizeof(uint32_t)) /* counters */ );
		memset (dispersion_counters_previous, 0, (dispersion_allocated_count * sizeof(uint32_t *) ) /* pointers */ +
			(dispersion_allocated_count * MY_MAX_ROUTES * sizeof(uint32_t)) /* counters */ );

		counters_basse_offset = (uint32_t *) (dispersion_counters + dispersion_allocated_count);
		counters_basse_offset_previous = (uint32_t *) (dispersion_counters_previous + dispersion_allocated_count);
		for (i = 0; i < dispersion_allocated_count; i++) 
		{
			dispersion_counters[i] = counters_basse_offset + (MY_MAX_ROUTES * i);
			dispersion_counters_previous[i] = counters_basse_offset_previous + (MY_MAX_ROUTES * i);
		}
	}
#endif

	return  1;
}

int MYRINET_num_counters()
{
	return count;
}

int read__myri_counters(mx_endpoint_t *mxport, uint32_t **counters_ptr) {
        mx_return_t ret;

        CHECK_GLOBALS();
        //printf("About to read counters\n");
        do {
/*            printf("Reading board counters %d (allocated %d) - board:%d (%d) - counters_ptr=%p\n", 
                            count, allocated_count, board_id, sizeof(board_id), *counters_ptr); */
            ret = mx_get_info(NULL, MX_COUNTERS_VALUES, &board_id, sizeof (board_id),
                              *counters_ptr, sizeof (uint32_t) * allocated_count);
                              //counters_ptr, sizeof (counters_ptr));
            if ((ret != MX_SUCCESS) && (errno != EBUSY)) {
              perror("get counters failed");
              mxmpi_abort(MX_FAILURE);
            }
        } while (ret);

        if (ret == MX_SUCCESS) {
                /* */
        }
        else {
                /* Insert event that counters could not be read?? --> no */
                fprintf(stderr,"Problem reading counters\n");
        }

        return 1;
}

int MYRINET_reset_counters() 
{
	CHECK_GLOBALS();

	read__myri_counters(&mxmpi.my_endpoint, &counters_previous);
	counters_previous_defined = 1;

#ifdef MX_MARENOSTRUM_API
	/**************  THIS CLEARS ALL  COUNTERS ************************/
	MYRINET_reset_routes();
#endif
	return 1;
}


#define CHECK_DIFF(xfield) ( (counters_previous_defined) ? ((counters[(xfield)])-(counters_previous[(xfield)])) : 0 )
#ifdef MX_MARENOSTRUM_API
#define CHECK_DISP_DIFF(mpirank,xfield) ( (dispersion_counters_previous_defined) ? ((dispersion_counters[(mpirank)][(xfield)])-(dispersion_counters_previous[(mpirank)][(xfield)])) : 0 )
#endif
#define VALUE(xfield) ( (counters[(xfield)]) )


int MYRINET_read_counters (int num_events, u_int32_t * values) 
{
	int event_counter = 0;

	num_events = MIN(num_events, MYRINET_num_counters());

	read__myri_counters(&mxmpi.my_endpoint, &counters);

#ifdef MPITRACE
	for (event_counter = 0; event_counter < num_events; event_counter ++) 
	{
		//if (phys_ports < 2) {
		if (strstr(&names[event_counter * MX_MAX_STR_LEN], "(Port 1)")) 
		{
			continue;
		}
		//}
		values[event_counter] = CHECK_DIFF(event_counter);
	}
#endif

#ifdef MX_PRINT
	for (event_counter = 0; event_counter < num_events; event_counter ++) 
	{
		//if (phys_ports < 2) {
		if (strstr(&names[event_counter * MX_MAX_STR_LEN], "(Port 1)")) 
		{
			continue;
		}
	//}
		printf("%34s: %10u (0x%x)\n", &names[event_counter * MX_MAX_STR_LEN], 
			(unsigned int)counters[event_counter], 
			(unsigned int)counters[event_counter]);
	}
#endif

	SWAP(counters, counters_previous);
	counters_previous_defined = 1;

	return event_counter;
}

int MYRINET_counters_labels (char *** avail_counters)
{
	int i;
	int event_counter;
	char ** counters_list;

	counters_list = (char **)malloc(MYRINET_num_counters() * sizeof(char *));

	event_counter = 0;
	for (i = 0; i < count; i ++)
	{
		//if (phys_ports < 2) {
		if (strstr(&names[i * MX_MAX_STR_LEN], "(Port 1)")) {
			continue;
		}
		//}
		counters_list[event_counter] = (char *)malloc(strlen(&names[ i * MX_MAX_STR_LEN ])+1);
		strcpy(counters_list[event_counter], (const char *)&names[ i * MX_MAX_STR_LEN ]);
		event_counter ++;
	}
	*avail_counters = counters_list;
	return event_counter;
}

#ifdef MX_MARENOSTRUM_API
inline int read__dispersion_myri_counters(mx_endpoint_t *mxport, uint32_t *dispersion_counters, int mpi_rank) {
	mx_return_t ret;

	ret = mx_get_dispersion_counters(*mxport, mxmpi.addrs[mpi_rank], dispersion_counters);
        /*
        {
                int i;
                for (i = 0; i < MY_MAX_ROUTES; i++) {
                        printf("%d:DispersionCounterRead = %d\n", mpi_rank, dispersion_counters[i]);
                }
        }
        */
#ifdef DO_CHECKS
        if ((ret != MX_SUCCESS) && (errno != EBUSY)) {
           perror("get dispersion counters failed");
           mxmpi_abort(MX_FAILURE);
        }
#endif
	return 1;
}

int MYRINET_num_routes()
{
	return MY_MAX_ROUTES;
}

int MYRINET_reset_routes ()
{
	int i; /* mpi_rank */

	CHECK_DISP_GLOBALS();

	for (i = 0; i < MPID_MyWorldSize; i ++) 
	{
		read__dispersion_myri_counters(&mxmpi.my_endpoint, dispersion_counters_previous[i], i);
	}
	dispersion_counters_previous_defined = 1;

	return 1;
}

int MYRINET_read_routes (int mpi_rank, int num_routes, uint32_t * values) {
	int event_counter = 0;

	num_routes = MIN(num_routes, MYRINET_num_routes());

	read__dispersion_myri_counters(&mxmpi.my_endpoint, dispersion_counters[mpi_rank], mpi_rank);

#ifdef MPITRACE
	for (event_counter = 0; event_counter < MY_MAX_ROUTES; event_counter ++) 
	{
		values[event_counter] = CHECK_DISP_DIFF(mpi_rank, event_counter);
	}
#endif

#ifdef MX_PRINT
	for (event_counter = 0; event_counter < MY_MAX_ROUTES; event_counter ++) 
	{
		uint32_t diff;
		diff = CHECK_DISP_DIFF(mpi_rank, event_counter);
		printf("DispersionCounter[from=%d][to=%d][route=%d]: %10u - %10u ; diff = %u\n",
			MPID_MyWorldRank, mpi_rank, event_counter,
			(unsigned int)dispersion_counters[mpi_rank][event_counter], 
			(unsigned int)dispersion_counters_previous[mpi_rank][event_counter],
			CHECK_DISP_DIFF(mpi_rank, event_counter));
	}
#endif

	SWAP(dispersion_counters[mpi_rank], dispersion_counters_previous[mpi_rank]);
	dispersion_counters_previous_defined = 1;
	
	return event_counter;
}
#endif

#ifdef MY_MAIN

#define COUNT 1024*1024*20 /*1024*1024*/
int send_array[COUNT];
int recv_array[COUNT];

int main (int argc, char *argv[]) {
    int rank,nprocs;
        double t1,t2;
        int i;

        MPI_Init (&argc, &argv);

        MPI_Comm_rank (MPI_COMM_WORLD, &rank);
        MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
        printf("I'm %d from %d processors\n", rank, nprocs);
	fflush(stdout);

        MYRINET_reset_counters();

        for (i = 0; i < 2; i++) {
				uint32_t counters[count];
                t1 = MPI_Wtime ();
                MYRINET_read_counters(count, counters);
                t2 = MPI_Wtime ();

                {
                        int dest_rank;
                        int source_rank;
#ifdef MX_MARENOSTRUM_API
						uint32_t dest_routes[MY_MAX_ROUTES];
						uint32_t source_routes[MY_MAX_ROUTES];
#endif
                        MPI_Status status;

                        source_rank = (rank == 0) ? nprocs - 1 : rank - 1;
                        dest_rank = (rank == (nprocs-1)) ? 0 : rank + 1;
#ifdef MX_MARENOSTRUM_API
    	            	MYRINET_read_routes(dest_rank, MY_MAX_ROUTES, dest_routes);
  	                	MYRINET_read_routes(source_rank, MY_MAX_ROUTES, source_routes);
#endif
                        MPI_Sendrecv (send_array, COUNT, MPI_INT, dest_rank, 1000, 
                                      recv_array, COUNT, MPI_INT, source_rank, 1000,
                                      MPI_COMM_WORLD, &status);
                        printf("First byte received: %d\n", recv_array[0]);
                }

        }

        printf("I spent %f seconds in reading the counters (WTime resolution is: %f)\n", t2-t1, MPI_Wtick());

        MPI_Barrier(MPI_COMM_WORLD);
        
        MPI_Finalize();


        return 1;
}

#endif
