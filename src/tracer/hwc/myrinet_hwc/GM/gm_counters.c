/*
 * This small MPI program tries to illustrate the use of the reading of the
 * gm counters within a small MPI app.
 *
 * COMPILE:
 *

mpicc -o mpi_read_counters mpi_read_counters.c -I/opt/osshpc/gm/include -I~/gm-2.0.21_MacOSX/include -I~/gm-2.0.21_MacOSX/drivers/linux/gm -lgm

 * ONLY WORKS WITH VERSION 2.0.21 - same version as compiled driver!!
 */

//#define DEBUG 1
//#define MY_MAIN 1
//#define MPITRACE
//#define GM_PRINT 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gm.h"
/* This program uses undocumented software interfaces that are not
   intended for use in code written outside Myricom.  Hence the
   following undocumented includes. */
#include "gm_internal.h"
#include "gm_enable_error_counters.h"
#include "gm_enable_debug_counters.h"
#include "gm_enable_packet_counters.h"

#ifdef MY_MAIN
#include "mpi.h"
#endif

#define MPITRACE

#include "gm_counters.h"

extern struct gm_port *gmpi_gm_port;



//#define MY_MAX_NODE (3*1024)
#define MY_MAX_NODE (6655)

/*
unsigned int space[(sizeof(struct gm_lanai_globals) +
		       ((MY_MAX_NODE+1) * sizeof(struct gm_connection))) / sizeof(int)];
unsigned int space_previous[(sizeof(struct gm_lanai_globals) +
		       ((MY_MAX_NODE+1) * sizeof(struct gm_connection))) / sizeof(int)];

struct gm_lanai_globals *globals = (struct gm_lanai_globals *) space;
struct gm_lanai_globals *globals_previous = (struct gm_lanai_globals *) space_previous;
*/

struct gm_lanai_globals *globals = NULL;
struct gm_lanai_globals *globals_previous = NULL;

static int globals_previous_defined = 0;

#define SWAP(x,y) { void *ptr; ptr = x; x = y; y = ptr; }

#define CHECK_GLOBALS() { if ( globals == NULL ) { MYRINET_start_counters(); } }

void MYRINET_start_counters() {
	int length;

	length = sizeof(struct gm_lanai_globals) + (MY_MAX_NODE * sizeof(struct gm_connection));
        
	if (globals == NULL) {
		if ((globals = malloc (length)) == NULL) {
			printf ("gm_counters: globals, error in malloc\n");
			gm_exit (GM_FAILURE);
		}
		if ((globals_previous = malloc (length)) == NULL) {
			printf ("gm_counters: globals_previous, error in malloc\n");
			gm_exit (GM_FAILURE);
		}
	}
}

gm_port_t *port1;

int read__myri_counters(struct gm_port *port, struct gm_lanai_globals *globals_ptr) {
	int length;
	gm_status_t status;

	/* 
 	 * Find "MAX" of offset + sizeof(data) to pass it on to 
	 * "length" parameter of _gm_get_globals
	 */

	//length = sizeof(struct gm_lanai_globals) + (MY_MAX_NODE * sizeof(struct gm_connection));
	length = sizeof(struct gm_lanai_globals); 
        
	CHECK_GLOBALS();
	status = _gm_get_globals(port, (gm_u8_t *)globals_ptr, length);
	//status = _gm_user_ioctl (port, GM_GET_GLOBALS, (gm_u8_t *)globals_ptr, length);
	if (status == GM_SUCCESS) {
/*
	if (GM_ENABLE_PACKET_COUNTERS) {
		printf("netsend_cnt      %u\n", gm_ntoh_u32 ((globals_ptr)->netsend_cnt));
		printf("netrecv_cnt      %u\n", gm_ntoh_u32 ((globals_ptr)->netrecv_cnt));
	}
	if (GM_ENABLE_ERROR_COUNTERS) {
		#define GM_ERROR_CNT(name,desc)										\
		printf ("%-40s %u\n", #name, gm_ntoh_u32 ((&globals)->name ## _error_cnt));
		GM_ERROR_COUNTERS
		#include "gm_error_counters.h"
	}
*/
	}
	else {
		/* Insert event that counters could not be read?? --> no */
		printf("Problem reading counters\n");
	}

	return 1;
}

void MYRINET_reset_counters() {

	/* clear counters if requested */

	/* 
	gm_status_t status;
	status = _gm_clear_counters(port);
	
	if (status != GM_SUCCESS)
	{
		gm_perror("error clearing counters", status);
		gm_exit(GM_FAILURE);
	}
	*/

	CHECK_GLOBALS();
	read__myri_counters(gmpi_gm_port, globals_previous);
	globals_previous_defined = 1;
}

int MYRINET_num_counters()
{
	int max_counters;

	max_counters = 0;
	if (GM_ENABLE_PACKET_COUNTERS)
	{
		max_counters += 2; /* netsend_cnt & netrecv_cnt */
	}
	if (GM_ENABLE_ERROR_COUNTERS) 
	{
		max_counters += GM_NUM_ERROR_COUNTERS;
	}
	return max_counters;
}

#define CHECK_DIFF(xfield) ( (globals_previous_defined) ? (gm_ntoh_u32((globals->xfield))-gm_ntoh_u32((globals_previous->xfield))) : 0 )

#define FIRST_EVENT 10000

#define MIN(x,y) ((x < y) ? x : y)

int MYRINET_read_counters(int num_events, u_int32_t * values) {
	int event_counter = 0;

	num_events = MIN(num_events, MYRINET_num_counters());

	read__myri_counters(gmpi_gm_port, globals);

#ifdef MPITRACE
	if (GM_ENABLE_PACKET_COUNTERS) {
		if (num_events > event_counter) 
		{
			values[event_counter++] = CHECK_DIFF(netsend_cnt) ;
		}
		if (num_events > event_counter) 
		{
			values[event_counter++] = CHECK_DIFF(netrecv_cnt) ;
		}
	}
	if (GM_ENABLE_ERROR_COUNTERS) {
#		undef GM_ERROR_CNT
#		define GM_ERROR_CNT(name,desc)                                   \
			if (num_events > event_counter)                              \
			{                                                            \
				values[event_counter++] = CHECK_DIFF(name ## _error_cnt) ; \
			}
		GM_ERROR_COUNTERS
#		include "gm_error_counters.h"
	}
#endif

#ifdef GM_PRINT
	if (GM_ENABLE_PACKET_COUNTERS) {
		printf("diff - netsend_cnt      %u\n", CHECK_DIFF(netsend_cnt));
		printf("diff - netrecv_cnt      %u\n", CHECK_DIFF(netrecv_cnt));
	}
	if (GM_ENABLE_ERROR_COUNTERS) {
#		undef GM_ERROR_CNT
#		define GM_ERROR_CNT(name,desc) \
			printf ("diff - %-40s %u\n", #name, CHECK_DIFF (name ## _error_cnt));
		GM_ERROR_COUNTERS
#		include "gm_error_counters.h"
	}
#endif

	SWAP(globals, globals_previous);
	globals_previous_defined = 1;

	return event_counter;
}

int MYRINET_counters_labels (char *** avail_counters) {
	int event_counter;
	char ** counters_list;

	counters_list = (char **)malloc(MYRINET_num_counters() * sizeof(char *));

	event_counter = 0;
	if (GM_ENABLE_PACKET_COUNTERS) {
		counters_list[event_counter] = (char *)malloc(strlen("netsend_cnt")+1);
		strcpy(counters_list[event_counter], (const char *)"netsend_cnt");
		event_counter ++;

		counters_list[event_counter] = (char *)malloc(strlen("netrecv_cnt")+1);
		strcpy(counters_list[event_counter], (const char *)"netrecv_cnt");
		event_counter ++;
	}
	if (GM_ENABLE_ERROR_COUNTERS) {
#		undef GM_ERROR_CNT
#		define GM_ERROR_CNT(name,desc)                                      \
			counters_list[event_counter] = (char *)malloc(strlen(#name)+1); \
            strcpy(counters_list[event_counter], (const char *)#name);      \
            event_counter ++;
		GM_ERROR_COUNTERS
#		include "gm_error_counters.h"
	}
	*avail_counters = counters_list;
	return event_counter;
}

#ifdef MY_MAIN

//#define COUNT 1024*1024*20 /*1024*1024*/
#define COUNT 1 /*1024*1024*/
int send[COUNT];
int recv[COUNT];

int main (int argc, char *argv[]) {
        int rank,nprocs;
        double t1,t2;
        int i;
		u_int32_t values[2];

        MPI_Init (&argc, &argv);
        MYRINET_reset_counters();

        for (i = 0 ; i < COUNT; i++) {

        }

        MPI_Comm_rank (MPI_COMM_WORLD, &rank);
        MPI_Comm_size (MPI_COMM_WORLD, &nprocs);

        printf("I'm %d from %d processors\n", rank, nprocs);

        for (i = 0; i < 2; i++) {
                t1 = MPI_Wtime ();
                MYRINET_read_counters (2, values);
                t2 = MPI_Wtime ();

                {
                        int dest_rank;
                        int source_rank;
                        MPI_Status status;

                        source_rank = (rank == 0) ? nprocs - 1 : rank - 1;
                        dest_rank = (rank == (nprocs-1)) ? 0 : rank + 1;

                        MPI_Sendrecv (send, COUNT, MPI_INT, dest_rank, 1000, 
                                      recv, COUNT, MPI_INT, source_rank, 1000,
                                      MPI_COMM_WORLD, &status);
                        printf("First byte received: %d\n", recv[0]);
                }

        }

        printf("I spent %f seconds in reading the counters (WTime resolution is: %f)\n", t2-t1, MPI_Wtick());

#ifdef GM_PRINT
        if (rank == 0) {
				char ** labels;
                MYRINET_counters_labels (&labels);
        }
#endif

        MPI_Barrier(MPI_COMM_WORLD);
        
        MPI_Finalize();


        return 1;
}

#endif
