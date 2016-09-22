
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

#include "common.h"

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif

#if defined(HAVE_BFD)

#include "debug.h"
#include "bfd_manager.h"
#include "object_tree.h"

static loadedModule_t *loadedModules = NULL;
static unsigned numLoadedModules = 0;

static bfd *default_bfdImage = NULL;
static asymbol **default_bfdSymbols = NULL;

void BFDmanager_init (void)
{
	bfd_init();
}

unsigned BFDmanager_numLoadedBinaries(void)
{
	return numLoadedModules;
}

loadedModule_t *BFDmanager_getLoadedModule (unsigned idx)
{
	if (loadedModules != NULL && idx < numLoadedModules)
		return &loadedModules[idx];
	else
		return NULL;
}

static void BFDmanager_loadBFDdata (char *file, bfd **image, asymbol ***symbols,
	unsigned *nDataSymbols, data_symbol_t **DataSymbols)
{
	bfd *bfdImage = NULL;
	asymbol **bfdSymbols = NULL;

	if (nDataSymbols)
		*nDataSymbols = 0;
	if (DataSymbols)
		*DataSymbols = NULL;

	/* Open the binary file in read-only mode */
	bfdImage = bfd_openr (file, NULL);
	if (bfdImage == NULL)
	{
		const char *errmsg = bfd_errmsg (bfd_get_error());
		fprintf (stderr, "mpi2prv: WARNING! Cannot open binary file '%s': %s.\n"
		                "         Addresses will not be translated into source code references\n",
		  file, errmsg);
		return;
	}

	/* Check the binary file format */
	if (!bfd_check_format (bfdImage, bfd_object))
	{
		const char *errmsg = bfd_errmsg( bfd_get_error() );
		fprintf (stderr, "mpi2prv: WARNING! Binary file format does not match for file '%s' : %s\n"
		                "         Addresses will not be translated into source code references\n",
		  file, errmsg);
	}

	/* Load the mini-Symbol Table */
	if (bfd_get_file_flags (bfdImage) & HAS_SYMS)
	{
		long symcount;
		size_t size = bfd_get_symtab_upper_bound (bfdImage);
		if (size > 0)
		{
#if defined(BFD_MANAGER_GENERATE_ADDRESSES)
			long s;
			unsigned nDataSyms = 0;
			data_symbol_t *DataSyms = NULL;
#endif

			bfdSymbols = (asymbol**) malloc (size);
			if (bfdSymbols == NULL)
				FATAL_ERROR ("Cannot allocate memory to translate addresses into source code references\n");

#if 0
			/* HSG This is supposed to be space-efficient, but showed some errors .... :( */
			symcount = bfd_read_minisymbols (bfdImage, FALSE, (PTR) bfdSymbols, &usize);
			if (symcount == 0) 
				symcount = bfd_read_minisymbols (bfdImage, TRUE, (PTR) bfdSymbols, &usize);
#else
			symcount = bfd_canonicalize_symtab (bfdImage, bfdSymbols);

# if defined(BFD_MANAGER_GENERATE_ADDRESSES)
			if (nDataSymbols && DataSymbols)
			{
				for (s = 0; s < symcount; s++)
				{
					symbol_info syminfo;
					bfd_get_symbol_info (bfdImage, bfdSymbols[s], &syminfo);
					if (((bfdSymbols[s]->flags & BSF_DEBUGGING) == 0) &&
					    (syminfo.type == 'R' || syminfo.type == 'r' || /* read-only data */
					    syminfo.type == 'B' || syminfo.type == 'b' || /* uninited data */
					    syminfo.type == 'G' || syminfo.type == 'g' || /* inited data */ 
					    syminfo.type == 'C')) /* common data*/
					{
						unsigned long long sz = 0;
						if (bfd_get_flavour(bfdImage) == bfd_target_elf_flavour)
							sz = ((elf_symbol_type*) bfdSymbols[s])->internal_elf_sym.st_size;

						DataSyms = (data_symbol_t*) realloc (DataSyms, sizeof(data_symbol_t)*(nDataSyms+1));
						if (DataSyms == NULL)
							FATAL_ERROR ("Cannot allocate memory to allocate data symbols\n");
						DataSyms[nDataSyms].name = strdup (syminfo.name);
						DataSyms[nDataSyms].address = (void*) syminfo.value;
						DataSyms[nDataSyms].size = sz;
						nDataSyms++;
					}
				}

				*nDataSymbols = nDataSyms;
				*DataSymbols = DataSyms;
			}
# endif
#endif

			if (symcount < 0) 
			{
				/* There aren't symbols! */
				const char *errmsg = bfd_errmsg( bfd_get_error() );
				fprintf(stderr, "mpi2prv: WARNING! Cannot read symbol table for file '%s' : %s\n"
		                "         Addresses will not be translated into source code references\n",
				  file, errmsg);
			}
		}
	}

	*image = bfdImage;
	*symbols = bfdSymbols;

#if defined(DEBUG)
	fprintf (stderr, "BFD file=%s bfdImage = %p bfdSymbols = %p\n",
	  file, bfdImage, bfdSymbols);
#endif

}

void BFDmanager_loadBinary (char *file, bfd **bfdImage, asymbol ***bfdSymbols,
	unsigned *nDataSymbols, data_symbol_t **DataSymbols)
{
	unsigned u, idx;

	/* Check for already loaded? If so, return preloaded values */
	for (u = 0; u < numLoadedModules; u++)
		if (strcmp (loadedModules[u].module, file) == 0)
		{
			*bfdImage = loadedModules[u].bfdImage;
			*bfdSymbols = loadedModules[u].bfdSymbols;
			return;
		}

	loadedModules = (loadedModule_t*) realloc (loadedModules,
	  (numLoadedModules+1)*sizeof(loadedModule_t));
	if (loadedModules == NULL)
		FATAL_ERROR("Cannot obtain memory to load a binary");

	idx = numLoadedModules;
	loadedModules[idx].module = strdup (file);
	if (loadedModules[idx].module == NULL)
		FATAL_ERROR("Cannot obtain memory to duplicate module name");

	BFDmanager_loadBFDdata (loadedModules[idx].module, &loadedModules[idx].bfdImage,
	  &loadedModules[idx].bfdSymbols, nDataSymbols, DataSymbols);
	*bfdImage = loadedModules[idx].bfdImage;
	*bfdSymbols = loadedModules[idx].bfdSymbols;
	numLoadedModules++;
}

void BFDmanager_loadDefaultBinary (char *file)
{
	BFDmanager_loadBFDdata (file, &default_bfdImage, &default_bfdSymbols,
		NULL, NULL);
}

bfd *BFDmanager_getDefaultImage (void)
{
	return default_bfdImage;
}

asymbol **BFDmanager_getDefaultSymbols (void)
{
	return default_bfdSymbols;
}

/** Find_Address_In_Section
 *
 * Localitza la direccio (pc) dins de la seccio ".text" del binari
 *
 * @param abfd
 * @param section
 * @param data
 *
 * @return No return value.
 */
static void BFDmanager_findAddressInSection (bfd * abfd, asection * section, PTR data)
{
#if HAVE_BFD_GET_SECTION_SIZE || HAVE_BFD_GET_SECTION_SIZE_BEFORE_RELOC
	bfd_size_type size;
#endif
	bfd_vma vma;
	BFDmanager_symbolInfo_t *symdata = (BFDmanager_symbolInfo_t*) data;

	if (symdata->found)
		return;

	if ((bfd_get_section_flags (abfd, section) & SEC_ALLOC) == 0)
		return;

	vma = bfd_get_section_vma (abfd, section);;

	if (symdata->pc < vma)
		return;

#if HAVE_BFD_GET_SECTION_SIZE
	size = bfd_get_section_size (section);
	if (symdata->pc >= vma + size)
		return;
#elif HAVE_BFD_GET_SECTION_SIZE_BEFORE_RELOC
	size = bfd_get_section_size_before_reloc (section);
	if (symdata->pc >= vma + size)
		return;
#else
	/* Do nothing? */
#endif

	symdata->found = bfd_find_nearest_line (abfd, section, symdata->symbols,
	  symdata->pc - vma, &symdata->filename, &symdata->function,
	  &symdata->line);
}

int BFDmanager_translateAddress (bfd *bfdImage, asymbol **bfdSymbols,
	void *address, char **function, char **file, int *line)
{
	BFDmanager_symbolInfo_t syminfo;
	char caddress[32];

#if defined(DEBUG)
	printf ("DEBUG: BFDmanager_translateAddress (%p, %p, address = %p)\n", bfdImage, bfdSymbols, address);
#endif

	syminfo.found = FALSE;

	if (bfdImage && bfdSymbols)
	{
		/* Convert the address into hexadecimal string format */
		sprintf (caddress, "%p", address);

		/* Prepare the query for BFD */
		syminfo.pc = bfd_scan_vma (caddress, NULL, 16);
		syminfo.symbols = bfdSymbols;

		/* Iterate through sections of the given bfd image */
		bfd_map_over_sections (bfdImage, BFDmanager_findAddressInSection, &syminfo);

		/* Found the symbol ? If so, copy the data */
		if (syminfo.found)
		{
			char *demangled = NULL;
			*file = (char*) syminfo.filename;
			*line = syminfo.line;

#if defined(HAVE_BFD_DEMANGLE)
			if (syminfo.function)
				demangled = bfd_demangle (bfdImage, syminfo.function, 0);

			if (demangled)
				*function = demangled;
			else
				*function = (char*) syminfo.function;
#else
			*function = (char*) syminfo.function;
#endif
		}
	}

#if defined(DEBUG)
	printf ("DEBUG: BFDmanager_translateAddress found? %d\n", syminfo.found);
	if (syminfo.found)
		printf ("DEBUG: BFDmanager_translateAddress file = %s function = %s line = %d\n", *file, *function, *line);
#endif

	return syminfo.found;
}

#endif /* HAVE_BFD */

#if defined(DEAD_CODE)

#if defined(USE_SYSTEM_CALL)
# ifdef HAVE_SYS_TYPES_H
#  include <sys/types.h>
# endif
# ifdef HAVE_SYS_WAIT_H
#  include <sys/wait.h>
# endif
# ifdef HAVE_UNISTD_H
#  include <unistd.h>
# endif
# ifdef HAVE_SYS_STAT_H
#  include <sys/stat.h>
# endif
# ifdef HAVE_FCNTL_H
#  include <fcntl.h>
# endif
#endif

int system_call_to_addr2line (char *binary, void *address,
	char **translated_function, char **translated_filename, int *translated_line)
{
	char *cmd[] = { "addr2line", "-f", "-e", binary, address, (char *)0 };
	pid_t pid;
	int status;
	int fd[2], null_fd;
	int matches;
	char addr2line_result[1024];
	char **tmp1 = NULL, **tmp2 = NULL;
	char caddress[16];

	/* Convert the address into hexadecimal string format */
	sprintf (caddress, "%p", address);

	pipe(fd);

	pid = fork();
	switch(pid)
	{
		case -1:
			perror("fork");
			exit(1);

		case 0: 
			close (1); /* close stdout */
			dup (fd[1]); /* make stdout same as fd[1] */

			/* Redirect stderr to /dev/null */
			null_fd = open("/dev/null", O_WRONLY);
			if (null_fd > 0)
			{
				close(2); /* close stderr */
				dup (null_fd);
			}

			close (fd[0]); /* we don't need this */
			execvp ("addr2line", cmd);
			break;
	}
	close (fd[1]); 
	read (fd[0], addr2line_result, sizeof(addr2line_result));
	wait(&status);

	matches = explode(addr2line_result, "\n", &tmp1);
	if (matches < 2)
		return FALSE;
	matches = explode(tmp1[1], ":", &tmp2);
	if (matches < 2)
		return FALSE;

	*translated_function = tmp1[0];
	*translated_filename = tmp2[0];
	*translated_line = atoi(tmp2[1]);

	if (!strcmp(translated_function, "??") && !strcmp(translated_filename, "??") && translated_line == 0)
		return FALSE;
	else
		return TRUE;
}

#endif 

