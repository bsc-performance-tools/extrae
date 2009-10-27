/*************************************************************************
 * Myricom MPICH-MX ch_mx backend                                        *
 * Copyright (c) 2003 by Myricom, Inc.                                   *
 * All rights reserved.                                                  *
 *************************************************************************/

#ifndef _mxmpi_h
#define _mxmpi_h

/* Used to output debug info */
#if 1
#define MXMPI_NOTE(S)
#else
#define MXMPI_NOTE(S) do {						\
  printf("[%d] ", MPID_MyWorldRank);					\
  printf S;								\
} while (0)
#endif

#include <assert.h>
#include <stdio.h>

#include "myriexpress.h"

#include "mpichconf.h"
#include "mpich-mpid.h"

#include "mxmpi_debug_checksum.h"

/* Defintions for network stuff */
#include <sys/types.h>
#include <netinet/in.h>

/*  For functions not yet implemented */
#define MXMPI_NOTIMPL() \
	fprintf(stderr, "Not yet implemented at %s:%d\n", __FILE__, __LINE__);\
	exit(1); 

#define MXMPI_MATCH_MASK_SRC    ((uint64_t)0xffffffffU << 32 | 0x0000ffffU)
#define MXMPI_MATCH_MASK_TAG    ((uint64_t)0xffffffffU << 32 | 0xffff0000U)
#define MXMPI_MATCH_MASK_SRC_TAG   ((uint64_t)0xffffffffU << 32)
#define MXMPI_MATCH(CONTEXT_ID,SRC,TAG) \
	       ((((uint64_t)(uint32_t)(CONTEXT_ID)) << 32)|\
                (((uint32_t)(SRC)&0xFFFF) << 16)|\
                ((uint32_t)(TAG)&0xFFFF))
#define MXMPI_EXTRACT_TAG(M) ((M) & 0xFFFF)
#define MXMPI_EXTRACT_SRC(M) (((M) >> 16) & 0xFFFF)

/*
 * Misc definitions
 */
#define MXMPI_INIT_TIMEOUT (3*60)	/* ms to wait for config */

struct mxmpi_var {

  /* socket addresses for job configuration */
  struct sockaddr_in master_addr;
  struct sockaddr_in slave_addr;

  /* Information about this endpoint */
  mx_endpoint_t my_endpoint;
  mx_endpoint_addr_t my_endpoint_addr;
  int mpd;                     /* true if run under MPD */
  uint32_t filter_value;       /* filter value for communicating */

  /* Information about the other endpoints */
  uint64_t *nic_ids;           /* NIC IDs */
  mx_endpoint_addr_t *addrs;   /* MX addresses */
  char **host_names;           /* names of machines */
  char **exec_names;           /* names of executables */
  unsigned int *mpi_pids;      /* PIDs of MPI processes */

  /* runtime variables */
  mx_return_t (*test_or_wait)(mx_endpoint_t endpoint,
			      mx_request_t *request,
			      mx_status_t *status,
			      uint32_t *result);
};

#define MX_INTERP 0
#if MX_INTERP
extern FILE *interp_file;
#endif


/* Global subroutine defs */
void mxmpi_abort(int);
mx_return_t mxmpi_wait_wrapper(mx_endpoint_t endpoint,
			       mx_request_t *request,
			       mx_status_t *status,
			       uint32_t *result);

/*
 * externals
 */
extern struct mxmpi_var mxmpi;

/* debug stuff */
#define MXMPI_DEBUG 1

#if MXMPI_DEBUG
#define mxmpi_debug_assert assert
#else
#define mxmpi_debug_assert(a) 
#endif

#endif /* _mxmpi_h */
