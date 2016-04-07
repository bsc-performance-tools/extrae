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

#ifndef BFD_MANAGER_H_INCLUDED
#define BFD_MANAGER_H_INCLUDED

#ifdef HAVE_BFD

#include <bfd.h>

#include "bfd_data_symbol.h"

#if defined(BFD_MANAGER_GENERATE_ADDRESSES)
# include "bfd_manager_extra.h"
#endif

typedef struct loadedModule_st
{
	char *module;
	bfd *bfdImage;
	asymbol **bfdSymbols;
} loadedModule_t;

/* These variables are used to pass information between
   translate_addresses and find_address_in_section.  */
typedef struct BFDmanager_symbolInfo_st
{
	bfd_vma pc;
	asymbol **symbols;
	const char *filename;
	const char *function;
	unsigned int line;
	bfd_boolean found;
} BFDmanager_symbolInfo_t;

void BFDmanager_init (void);
unsigned BFDmanager_numLoadedBinaries (void);
loadedModule_t *BFDmanager_getLoadedModule (unsigned idx);
void BFDmanager_loadBinary (char *file, bfd **bfdImage, asymbol ***bfdSymbols,
	unsigned *nDataSymbols, data_symbol_t **DataSymbols);
int BFDmanager_translateAddress (bfd *bfdImage, asymbol **bfdSymbols, void *address,
	char **function, char **file, int *line);

void BFDmanager_loadDefaultBinary (char *file);
bfd *BFDmanager_getDefaultImage (void);
asymbol **BFDmanager_getDefaultSymbols (void);

#endif /* HAVE_BFD */

#endif /* BFD_MANAGER_H_INCLUDED */
