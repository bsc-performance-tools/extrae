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

#ifndef BFD_MANAGER_EXTRA_H_INCLUDED
#define BFD_MANAGER_EXTRA_H_INCLUDED

/* NOTE:
   The structures below are taken from binutils 2.23.2 file bfd/elf-bfd.h
*/

/* Information held for an ELF symbol.  The first field is the
   corresponding asymbol.  Every symbol is an ELF file is actually a
   pointer to this structure, although it is often handled as a
   pointer to an asymbol.  */

struct elf_internal_sym {
  bfd_vma   st_value;       /* Value of the symbol */
  bfd_vma   st_size;        /* Associated symbol size */
  unsigned long st_name;        /* Symbol name, index in string tbl */
  unsigned char st_info;        /* Type and binding attributes */
  unsigned char st_other;       /* Visibilty, and target specific */
  unsigned char st_target_internal; /* Internal-only information */
  unsigned int  st_shndx;       /* Associated section index */
};


/* Information held for an ELF symbol.  The first field is the
   corresponding asymbol.  Every symbol is an ELF file is actually a
   pointer to this structure, although it is often handled as a
   pointer to an asymbol.  */

typedef struct
{
	/* The BFD symbol.  */
	asymbol symbol;

	/* ELF symbol information.  */
	struct elf_internal_sym internal_elf_sym;

	/* Backend specific information.  */
	union
	{
		unsigned int hppa_arg_reloc;
		void *mips_extr;
		void *any;
	}
	tc_data;

	/* Version information.  This is from an Elf_Internal_Versym
	  structure in a SHT_GNU_versym section.  It is zero if there is no
	  version information.  */
	unsigned short version;

} elf_symbol_type;

#endif /* BFD_MANAGER_EXTRA_H_INCLUDED */
