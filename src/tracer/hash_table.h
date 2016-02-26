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

#include <config.h>

#ifndef _HASH_TABLE_H
#define _HASH_TABLE_H

#if !defined(MPI_SUPPORT) /* This shouldn't be compiled if MPI is not used */
# error "This should not be compiled outside MPI bounds"
#endif

#include "common.h"

#ifdef HAVE_MPI_H
# include <mpi.h>
#endif

#if !defined(TRUE)
# define TRUE  1
#endif

#if !defined(FALSE)
# define FALSE 0
#endif

/* Hash macros */
#define HASH_NULL (-1)
#define HASH_FREE (-2)
#define HASH_TABLE_SIZE 458879
#define HASH_OVERFLOW_SIZE ((HASH_TABLE_SIZE*15)/100)
#define HASH_FUNCTION(x)   (((uintptr_t)(x))%HASH_TABLE_SIZE)

/*
 * Some prime numbers for HASH_TABLE_SIZE
 *
 *         5       11       19       31      127
 *       211      383      631      887     1151
 *      1399     1663     1913     2161     2423
 *      2687     2939     3191     3449     3709
 *      3967     4219     4463     4733     4987
 *      5237     5503     5749     6011     6271
 *      6521     6781     7039     7283     7549
 *      7793     8059     8317     8573     8831
 *      9067     9343     9587     9851    10111
 *     10357    10613    10867    11131    11383
 *     11633    11903    12157    12413    12671
 *     12923    13183    13421    13693    13933
 *     14207    14461    14717    14969    15227
 *     15473    15739    15991    16253    16493
 *     18553    20599    22651    24697    26737
 *     28793    30841    32887    34939    36979
 *     39023    41081    43133    45181    47221
 *     49279    51307    53359    55411    57467
 *     59513    61561    63611    65657    82039
 *     98429   114809   131171   147583   163927
 *    180347   196727   213119   229499   245881
 *    262271   327799   393331   458879   524413
 *    589933   655471   721013   786553   852079
 *    917629   983153
 */

/* Hash table type definitions */

typedef struct
{
  MPI_Request key;                /* Hash key */
  MPI_Group group;                /* Allocated remote group of comm. GROUP_NULL => COMM_WORLD */
  MPI_Comm commid;                /* Communicator identifier */
	int partner;                    /* MPI p2p partner */
	int tag;                        /* MPI p2p tag */
	int size;                       /* MPI p2p size */
} hash_data_t;

typedef struct
{
  int ovf_link;                 /* First cell in overflow list. */
  /* HASH_FREE: free & no overflow cells */
  /* HASH_NULL: used & no overflow cells */
  /* >= 0:      used & overflow cells    */
  hash_data_t data;             /* User data */
} hash_tbl_t;

typedef struct
{
  int next;                     /* Overflow links */
  hash_data_t data;             /* User data */
} hash_ovf_t;

typedef struct
{
  hash_tbl_t table[HASH_TABLE_SIZE];
  hash_ovf_t overflow[HASH_OVERFLOW_SIZE];
  int ovf_free;                 /* First overflow free */
} hash_t;

void hash_init (hash_t * hash);
int hash_add (hash_t * hash, const hash_data_t * data);
hash_data_t *hash_search (const hash_t * hash, MPI_Request key);
int hash_remove (hash_t * hash, MPI_Request key);

#endif
