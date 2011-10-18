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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#ifdef HAVE_BFD

#undef USE_SYSTEM_CALL /* Define this to call system's addr2line command (use for debug purposes!) */

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_BFD_H
# include <bfd.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_LIBGEN_H
# include <libgen.h>
#endif
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

#if defined(NEED_ERRNO_LOCATION_PATCH)
/* On some machines (BG/L for instance) this symbol is undefined but required by
   BFD library. */
int *__errno_location(void)
{
	return 0;
}   
#endif

#include "events.h"
#include "addr2info.h"
#include "labels.h"
#include "misc_prv_semantics.h"
#include "addr2info_hashcache.h"

static void Read_SymTab (void);
static void AddressTable_Initialize (void);
static void Translate_Address (UINT64 address, char ** funcname, char ** filename, int * line);
static void Find_Address_In_Section (bfd * abfd, asection * section, PTR data);
static int  AddressTable_Insert (UINT64 address, int event_type, char * funcname, char * filename, int line);

struct address_table  * AddressTable  [COUNT_ADDRESS_TYPES]; /* Addresses translation table     */
struct function_table * FunctionTable [COUNT_ADDRESS_TYPES]; /* Function name translation table */

bfd * abfd;                          /* Binary file descriptor      */
static bfd_vma pc;                   /* BFD address representation  */
static asymbol ** symtab;            /* Binary symbol table         */
asection * section;                  /* Binary .text section        */

#if defined(USE_SYSTEM_CALL)
char *BinaryName = NULL;
#endif

/* Addresses will be translated into function, file and line if set to TRUE */
int Tables_Initialized = FALSE;
int Translate_Addresses = FALSE;

/* These variables are global to maintain backwards compatibility with the "map_over_sections" method */
static const char * translated_filename;
static const char * translated_funcname;
static unsigned int translated_line;
static int address_found;

/* Determine whether to write the translation tables into the PCF file */
#define A2I_MPI      0
#define A2I_OMP      1
#define A2I_UF       2
#define A2I_SAMPLE   3
#define A2I_CUDA     4
#define A2I_LAST     5
int Address2Info_Labels[A2I_LAST];

#define UNRESOLVED_ID 0
#define NOT_FOUND_ID 1

static int Address2Info_Sort_routine(const void *p1, const void *p2)
{
	struct address_info *a1 = (struct address_info*) p1;
	struct address_info *a2 = (struct address_info*) p2;

	/* Sort by filename, line and address */
	if (strcmp (a1->file_name, a2->file_name) == 0)
	{
		if (a1->line == a2->line)
		{
			if (a1->address == a2->address)
				return 0;
			else if (a1->address < a2->address)
				return -1;
			else
				return 1;
		}
		else if (a1->line < a2->line)
			return -1;
		else
			return 1;
	}
	else 
		return strcmp (a1->file_name, a2->file_name);
}

void Address2Info_Sort (int unique_ids)
{
	/* Sort identifiers and also skip UNRESOLVED and NOT_FOUND */
	if (unique_ids)
	{
		void *base = (void*) &(AddressTable[UNIQUE_TYPE]->address[2]);

		qsort (base, AddressTable[UNIQUE_TYPE]->num_addresses-2,
			sizeof(struct address_info), Address2Info_Sort_routine);
	}
	else
	{
		void *base = (void*) &(AddressTable[OUTLINED_OPENMP_TYPE]->address[2]);
		qsort (base, AddressTable[OUTLINED_OPENMP_TYPE]->num_addresses-2,
			sizeof(struct address_info), Address2Info_Sort_routine);

		base = (void*) &(AddressTable[MPI_CALLER_TYPE]->address[2]);
		qsort (base, AddressTable[MPI_CALLER_TYPE]->num_addresses-2,
			sizeof(struct address_info), Address2Info_Sort_routine);

		base = (void*) &(AddressTable[SAMPLE_TYPE]->address[2]);
		qsort (base, AddressTable[SAMPLE_TYPE]->num_addresses-2,
			sizeof(struct address_info), Address2Info_Sort_routine);

		base = (void*) &(AddressTable[USER_FUNCTION_TYPE]->address[2]);
		qsort (base, AddressTable[USER_FUNCTION_TYPE]->num_addresses-2,
			sizeof(struct address_info), Address2Info_Sort_routine);

		base = (void*) &(AddressTable[CUDAKERNEL_TYPE]->address[2]);
		qsort (base, AddressTable[CUDAKERNEL_TYPE]->num_addresses-2,
			sizeof(struct address_info), Address2Info_Sort_routine);
	}

	/* Cached entries are now invalid as everything gets resorted */
	Addr2Info_HashCache_Clean();
}

/** Address2Info_Initialized
 *
 * Check if the translation system is correctly initialized.
 * 
 * @param None
 *
 * @return Returns if Address2Info_Initialize has been succesfully called.
 */
int Address2Info_Initialized (void)
{
	return Translate_Addresses;
}

/** Address2Info_Initialize
 *
 * Initialize the memory addresses translation module.
 * 
 * @param binary
 *
 * @return No return value.
 */
void Address2Info_Initialize (char * binary)
{
	int type;
	char ** matching;

	Translate_Addresses = FALSE;

	/* Initialize the memory addresses translation table */
	AddressTable_Initialize();

	/* Add Unresolved and Not Found references now, so they always get
	   the same ID */
	for (type = 0; type < COUNT_ADDRESS_TYPES; type++)
	{
		/* Add fake address 0x0 to link "Unresolved" functions */
		AddressTable_Insert (UNRESOLVED_ID, type, ADDR_UNRESOLVED, ADDR_UNRESOLVED, 0);

		/* Add fake address 0x1 to link "Address not found" functions */
		AddressTable_Insert (NOT_FOUND_ID, type, ADDR_NOT_FOUND, ADDR_NOT_FOUND, 0);
	}

#if defined(USE_SYSTEM_CALL)
	/* Save the name of the application binary to pass it to the addr2line command */
	BinaryName = (char *)malloc((strlen(binary) + 1) * sizeof(char));
	if (BinaryName == NULL)
	{
		fprintf (stderr, "mpi2prv: Fatal error! Cannot allocate memory for binary name when doing a syscall!\n");
		exit (-1);
	}
	strcpy(BinaryName, binary);
#endif

	/* If no binary has been specified addresses won't be translated */
	if ((binary == (char *)NULL) || (strlen(binary) == 0)) 
		return;

	/* Initialize BFD libraries */
	bfd_init();

	/* Open the binary file in read-only mode */
	abfd = bfd_openr (binary, (char *)TARGET);
	if (abfd == NULL)
	{
		const char *errmsg = bfd_errmsg( bfd_get_error() );
		fprintf(stderr, "mpi2prv: WARNING! Cannot open binary file '%s': %s.\n"
		                "         Addresses will not be translated into source code references\n"
		                , binary, errmsg);
		return;
	}

	/* Check the binary file format */
	if (!bfd_check_format_matches (abfd, bfd_object, &matching))
	{
		const char * errmsg = bfd_errmsg( bfd_get_error() );
		fprintf(stderr, "mpi2prv: Binary file format doesn't match: %s\n", errmsg);
		exit(1);
	}

	/* Load the Symbol Table */
	Read_SymTab();

	/* Retrieve the binary code section (.text) */
	section = bfd_get_section_by_name(abfd, CODE_SECTION);
	if (section == NULL) 
	{
		const char * errmsg = bfd_errmsg( bfd_get_error() );
		fprintf(stderr, "mpi2prv: Section %s cannot be found: %s\n", CODE_SECTION, errmsg);
		exit(1);
	}

	/* Initialize the hash cache */
	Addr2Info_HashCache_Initialize();

	/* The addresses translation module has been successfully initialized */
	Translate_Addresses = TRUE;
}

/** Read_SymTab
 *
 * Load to memory the symbol table of the open binary
 *
 * @return No return value.
 */
static void Read_SymTab (void)
{
	long         symcount;
	unsigned int size;

	if ((bfd_get_file_flags (abfd) & HAS_SYMS) == 0) 
	{
		return;
	}
	symcount = bfd_read_minisymbols (abfd, FALSE, (PTR) &symtab, &size);
	if (symcount == 0) 
	{
		symcount = bfd_read_minisymbols (abfd, TRUE /* dynamic */, (PTR) &symtab, &size);
	}
	if (symcount < 0) 
	{
		/* There aren't symbols! */
		const char *errmsg = bfd_errmsg( bfd_get_error() );
		fprintf(stderr, "Error reading symbol table (bfd_read_minisymbols < 0): %s\n", errmsg);
		exit(1);
	}
/*	fprintf(stderr, "Number of symbols in SymTab: %d\n", symcount); */
}

/** AddressTable_Initialize
 *
 * Initialize the addresses translation table structures.
 *
 * @return No return value.
 */
static void AddressTable_Initialize (void)
{
	int type;

	for (type=0; type < COUNT_ADDRESS_TYPES; type++)
	{
		AddressTable[type] = (struct address_table *)malloc(sizeof(struct address_table));
		if (AddressTable[type] == NULL)
		{
			fprintf (stderr, "mpi2prv: Fatal error! Cannot allocate memory for AddressTable[type=%d]\n", type);
			exit (-1);
		}
		AddressTable[type]->address = NULL;
		AddressTable[type]->num_addresses = 0;

		FunctionTable[type] = (struct function_table *)malloc(sizeof(struct function_table));
		if (FunctionTable[type] == NULL)
		{
			fprintf (stderr, "mpi2prv: Fatal error! Cannot allocate memory for FunctionTable[type=%d]\n", type);
			exit (-1);
		}
		FunctionTable[type]->function = NULL;
		FunctionTable[type]->address_id = NULL;
		FunctionTable[type]->num_functions = 0;
	}
	Tables_Initialized = TRUE;
}

/** Address2Info_AddSymbol
 *
 * Add a symbol (address, name, filename and line tuple) in the translation
 * table of the specified address_type 
 */
int Address2Info_AddSymbol (UINT64 address, int addr_type, char * funcname,
	char * filename, int line)
{
	return AddressTable_Insert (address, addr_type, strdup(funcname), strdup(filename), line);
}

/** Address2Info_Translate
 *
 * 
 * Si l'adreca demanada no estava ja a la taula d'adreces, la tradueix pel nom
 * de la funcio i el numero de linea de codi corresponents, li assigna un nou
 * identificador i afegeix una nova entrada a la taula
 * 
 * @param address
 * @param query
 *
 * @return
 */
UINT64 Address2Info_Translate(UINT64 address, int query, int uniqueID)
{
	UINT64 caller_address;
	int addr_type;
	int i;
	int already_translated;
	int line;
	char * funcname;
	char * filename;
	int line_id = 0;
	int function_id = 0;
	UINT64 result;
	int found;


/* address es la direccion de retorno por donde continuara ejecutandose el
 * codigo despues de cada CALL a una rutina MPI. En arquitecturas x86, despues
 * de la llamada se desapilan los parametros y se recogen los resultados. Asi
 * pues, la @ de retorno apunta una instruccion de codigo maquina que se refiere
 * a la misma instruccion de invocacion del codigo de usuario, p.e:
 *
 *           mi_funcion(1,2,3);  =====>  add ... (apilar parametros)
 *                                       call 0x... (invocar funcion)
 *                     @ RETORNO ----->  sub ... (despilar parametros)
 *                                       ... (recoger resultado)
 *
 * Al traducir la direccion de cualquiera de las instrucciones maquinas de una misma
 * instruccion del codigo fuente, la linea / fichero obtenida es la misma. Por tanto, al
 * traducir directamente la @ obtenida por el backtrace, identificamos la linea de
 * la llamada en el codigo. EXCEPCION: Cuando la funcion no recibe ni recoge parametros,
 * la @ de retorno apunta directamente a la siguiente instruccion del codigo (p.e. MPI_Finalize)
 *
 * En arquitecturas POWER4, los parametros se pasan por REGISTRO. Despues de la llamada
 * a la funcion no se realizan mas operaciones, de modo que la @ de retorno apunta a una
 * instruccion maquina que se refiere a la siguiente instruccion del codigo fuente. Al
 * traducir esta direccion, obtenemos la SIGUIENTE linea donde se encuentra la llamada
 * MPI.
 *
 * Restando 1 a la direccion de retorno, pasamos a apuntar los bits intermedios de
 * la instruccion anterior, que es el CALL. El mecanismo de traduccion permite indicar
 * un octeto intermedio de una instruccion, y de esta forma obtendremos la linea correcta en ambos casos.
 *
 */

	/* Enable vars to generate labels in the PCF */
	switch (query)
	{
		case ADDR2MPI_FUNCTION:
		case ADDR2MPI_LINE:
			Address2Info_Labels[A2I_MPI] = TRUE;
			caller_address = address - 1;
			addr_type = uniqueID?UNIQUE_TYPE:MPI_CALLER_TYPE;
			break;
		case ADDR2OMP_FUNCTION:
		case ADDR2OMP_LINE:
			Address2Info_Labels[A2I_OMP] = TRUE;
			caller_address = address;
			addr_type = uniqueID?UNIQUE_TYPE:OUTLINED_OPENMP_TYPE;
			break;
		case ADDR2CUDA_FUNCTION:
		case ADDR2CUDA_LINE:
			Address2Info_Labels[A2I_CUDA] = TRUE;
			caller_address = address-1;
			addr_type = uniqueID?UNIQUE_TYPE:CUDAKERNEL_TYPE;
			break;
		case ADDR2UF_FUNCTION:
		case ADDR2UF_LINE:
			Address2Info_Labels[A2I_UF] = TRUE;
			caller_address = address-1;
			addr_type = uniqueID?UNIQUE_TYPE:USER_FUNCTION_TYPE;
			break;
		case ADDR2SAMPLE_FUNCTION:
		case ADDR2SAMPLE_LINE:
			Address2Info_Labels[A2I_SAMPLE] = TRUE;
			caller_address = address;
			addr_type = uniqueID?UNIQUE_TYPE:SAMPLE_TYPE;
			break;
		default:
			return address;
	}

	/* If subsystem is not enabled or address is 0, return */
	if (!Translate_Addresses || address == 0)
	{
		return address;
	}

/* Traduciremos caller_address para obtener la linea exacta del codigo, pero
 * escribiremos address en el PCF.
 * En un objdump y utilidades similares, no podemos buscar
 * la caller_address, que apunta a un octeto intermedio de una instruccion,
 * mientras que si encontraremos address, que es una direccion base (de la siguiente
 * instruccion).
 *
 * Por el momento no intentamos obtener la direccion base de la instruccion de call
 * por la diversidad de tamanyo de instrucciones en las distintas maquinas
 */

	found = Addr2Info_HashCache_Search (address, &line_id, &function_id);

	if (!found)
	{
		/* Search if address has been translated (search [1]) */
		already_translated = FALSE;
		for (i = 0; i < AddressTable[addr_type]->num_addresses; i++)
		{
			if (AddressTable[addr_type]->address[i].address == address)
			{
				already_translated = TRUE;
				function_id = AddressTable[addr_type]->address[i].function_id;
				line_id = i;
				break;
			}
		}
	}
	else
		already_translated = TRUE;

	if (!already_translated) 
	{
		/* Translate address into function, file and number line */
		Translate_Address (caller_address, &funcname, &filename, &line);

		/* Samples can be taken anywhere in the code. It can happen that two
		   samples at the different addresses refer to the same combination of
		   function/file/line... We search again if this combination has been
		   already translated */
		if (ADDR2SAMPLE_FUNCTION == query || ADDR2SAMPLE_LINE == query)
		{
			for (i = 0; i < AddressTable[addr_type]->num_addresses; i++)
			{
				if (AddressTable[addr_type]->address[i].line == line &&
				    strcmp (AddressTable[addr_type]->address[i].file_name, filename) == 0)
				{
					already_translated = TRUE;
					function_id = AddressTable[addr_type]->address[i].function_id;
					line_id = i;
					break;
				}
			}
		}

		if (strcmp (ADDR_UNRESOLVED, funcname) == 0 || strcmp (ADDR_UNRESOLVED, filename) == 0)
		{
			function_id = UNRESOLVED_ID;
			line_id = UNRESOLVED_ID;
			already_translated = TRUE;
		}
		else if (strcmp(ADDR_NOT_FOUND, funcname) == 0 || strcmp (ADDR_NOT_FOUND, filename) == 0)
		{
			function_id = NOT_FOUND_ID;
			line_id = NOT_FOUND_ID;
			already_translated = TRUE;
		}

		/* Again, if not found, insert into the translation table */
		if (!already_translated)
		{
			line_id = AddressTable_Insert(address, addr_type, funcname, filename, line);
			function_id = AddressTable[addr_type]->address[line_id].function_id;
		}
	}

	if (!found)
		Addr2Info_HashCache_Insert (address, line_id, function_id);

	result = 0;
	switch(query)
	{
		case ADDR2CUDA_FUNCTION:
		case ADDR2SAMPLE_FUNCTION:
		case ADDR2MPI_FUNCTION:
		case ADDR2UF_FUNCTION:
		case ADDR2OMP_FUNCTION:
			result = function_id + 1;
			break;
		case ADDR2CUDA_LINE:
		case ADDR2UF_LINE:
		case ADDR2SAMPLE_LINE:
		case ADDR2OMP_LINE:
		case ADDR2MPI_LINE:
			result = line_id + 1;
			break;
	}

	return result;
}

/** AddressTable_Insert
 *
 * @param address
 * @param funcname
 * @param filename
 * @param line
 * 
 * @return 
 */
static int AddressTable_Insert(UINT64 address, int addr_type, char * funcname, char * filename, int line) 
{
	int i; 
	int found;
	int new_address_id;
	struct address_table  * AddrTab;
	struct function_table * FuncTab;
	int function_id = 0;

	AddrTab = AddressTable[addr_type];
	FuncTab = FunctionTable[addr_type];

	new_address_id = AddrTab->num_addresses ++;
	AddrTab->address = (struct address_info *) realloc (
		AddrTab->address,
		AddrTab->num_addresses * sizeof(struct address_info)
	);
	if (NULL == AddrTab->address)
	{
		fprintf (stderr, "mpi2prv: Error! Cannot reallocate memory for AddressTable\n");
		exit (-1);
	}
	
	AddrTab->address[new_address_id].address = address;
	AddrTab->address[new_address_id].file_name = filename;
	AddrTab->address[new_address_id].line = line;

	/* Search for the function id */
	found = FALSE;
	for (i=0; i < FuncTab->num_functions; i++) 
	{
		if (strcmp(funcname, FuncTab->function[i]) == 0) 
		{
			found = TRUE;
			function_id = i;
			break;
		}
	}

	if (!found) 
	{
		function_id = FuncTab->num_functions ++;
		FuncTab->function = (char **) realloc (
			FuncTab->function, 
			FuncTab->num_functions * sizeof(char*)
		);
		if (NULL == FuncTab->function)
		{
			fprintf (stderr, "mpi2prv: Error! Cannot reallocate memory for function-identifiers table in FuncTab\n");
			exit (-1);
		}
		FuncTab->address_id = (UINT64*) realloc (
			FuncTab->address_id,
			FuncTab->num_functions * sizeof (UINT64)
		);
		if (NULL == FuncTab->address_id)
		{
			fprintf (stderr, "mpi2prv: Error! Cannot reallocate memory for address-identifiers table in FuncTab\n");
			exit (-1);
		}
		FuncTab->function[function_id] = funcname;
		FuncTab->address_id[function_id] = new_address_id;
	}
	AddrTab->address[new_address_id].function_id = function_id;

	return new_address_id;
}

#if defined(USE_SYSTEM_CALL)
int system_call_to_addr2line(char *binary, char *address)
{
	char *cmd[] = { "addr2line", "-f", "-e", binary, address, (char *)0 };
	pid_t pid;
	int status;
	int fd[2], null_fd;
	int matches;
	char addr2line_result[1024];
	char **tmp1 = NULL, **tmp2 = NULL;

	pipe(fd);

	pid = fork();
	switch(pid)
	{
		case -1:
			perror("fork");
			exit(1);
		case 0: 
			close(1); /* close stdout */
			dup(fd[1]); /* make stdout same as fd[1] */

			/* Redirect stderr to /dev/null */
			null_fd = open("/dev/null", O_WRONLY);
			if (null_fd > 0)
			{
				close(2); /* close stderr */
				dup(null_fd);
			}

			close(fd[0]); /* we don't need this */

			execvp("addr2line", cmd);
			break;
	}
	close(fd[1]); 
	read (fd[0], addr2line_result, sizeof(addr2line_result));
	wait(&status);

/*	fprintf(stderr, "ADDR2LINE_RESULT: %s\n", addr2line_result); */

	matches = explode(addr2line_result, "\n", &tmp1);
	if (matches < 2) {
		return 0;
	}
	matches = explode(tmp1[1], ":", &tmp2);
	if (matches < 2) {
		return 0;
	}

	translated_funcname = tmp1[0];
	translated_filename = tmp2[0];
	translated_line = atoi(tmp2[1]);

/*  fprintf(stderr, "[EXPLODED] func: %s file: %s line: %d", translated_funcname, translated_filename, translated_line); */
	if (!strcmp(translated_funcname, "??") && !strcmp(translated_filename, "??") && translated_line == 0)
		return 0;
	else
		return 1;
}
#endif /* USE_SYSTEM_CALL */

/** Translate_Address
 *
 * Tradueix l'adreca codificada al string address en format hexadecimal, al
 * nom de la funcio a la que correspon i el numero de linea del fitxer font
 * on es troba la instruccio
 *
 * @param address
 * @param [I/O] funcname
 * @param [I/O] filename
 * @param [I/O] line
 *
 * @return No return value.
 */
static void Translate_Address (UINT64 address, char ** funcname, char ** filename, int * line) {
	int resolved_function = FALSE; 
	int resolved_file = FALSE;
	char caddress[MAX_ADDR_LENGTH];
	char * file_basename;

	/* Convert the address into hexadecimal string format */
#if SIZEOF_LONG == 8
	sprintf(caddress, "0x%lx", address);
#elif SIZEOF_LONG == 4
	sprintf(caddress, "0x%llx", address);
#endif

	if (!Translate_Addresses) 
	{
		*funcname = ADDR_UNRESOLVED;
		*filename = ADDR_UNRESOLVED;
		*line = 0;
		return;
	}

	/* Convert the string into the BFD internal address representation format */
	pc = bfd_scan_vma (caddress, NULL, 16);

	address_found = FALSE;

#if defined(USE_SYSTEM_CALL)
	address_found = system_call_to_addr2line(BinaryName, caddress);
#else
	/*
	 *  Aquest metode itera sobre totes les seccions del binari, i
	 *  aplica la funcio "Find_Address_In_Section" dins de cada una
 	 */
	bfd_map_over_sections (abfd, Find_Address_In_Section, (PTR) NULL);
	 
	/* Some systems define routines (or trampolines to routines) in other sections
	   than .text (p.e. Linux / PPC has trampolines in the .DATA section

     The symbol we're looking for has to be located inside the ".text" section
	   of the binary. */
  /* Find_Address_In_Section(abfd, section, (PTR) NULL); */
#endif

	if (!address_found) 
	{
		/* The address has not been found in the binary */
		*filename = ADDR_NOT_FOUND;
		*funcname = ADDR_NOT_FOUND;
		*line = 0;
	}
	else 
	{
		*line = translated_line;

		if (translated_funcname == (char *)NULL) 
		{
			*funcname = ADDR_UNRESOLVED;
		}
		else 
		{
			resolved_function = TRUE;
			COPY_STRING(translated_funcname, *funcname);
		}

		if (translated_filename == (char *)NULL) 
		{
			*filename = ADDR_UNRESOLVED;
		}
		else 
		{
			resolved_file = TRUE;
			file_basename = basename((char *)translated_filename);
			COPY_STRING(file_basename, *filename);
		}
	}
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
static void Find_Address_In_Section (bfd * abfd, asection * section, PTR data)
{
	bfd_vma       vma;
#if defined(HAVE_BFD_GET_SECTION_SIZE) || defined(HAVE_BFD_GET_SECTION_SIZE_BEFORE_RELOC)
	bfd_size_type size;
#endif

	UNREFERENCED_PARAMETER(data);

	if (address_found) return;

	/* En cas d'iterar sobre totes les seccions amb "bfd_map_over_sections",
	 * i nomes busquem una d'especifica
	 *
	 * if (strcmp(section->name, SECTION) != 0) return;
	 */

	if ((bfd_get_section_flags (abfd, section) & SEC_ALLOC) == 0) return;

	vma = bfd_get_section_vma (abfd, section);

	/* Comprovem que l'adreca estigui dins de la seccio */
	if (pc < vma) return;

#if defined(HAVE_BFD_GET_SECTION_SIZE)
	size = bfd_get_section_size (section);
	if (pc >= vma + size) return;
#elif defined(HAVE_BFD_GET_SECTION_SIZE_BEFORE_RELOC)
	size = bfd_get_section_size_before_reloc (section);
	if (pc >= vma + size) return;
#else
    /* Do nothing? */
#endif

	address_found = bfd_find_nearest_line (abfd, section, symtab, pc - vma /*Offset dins seccio*/, &translated_filename, &translated_funcname, &translated_line);
/* Si es troba un simbol que coincideix amb la direccio en aquesta seccio,
 * filename, funcname and line actualitzen els seus valors per referencia
 */
}

void Address2Info_Write_MPI_Labels (FILE * pcf_fd, int uniqueid)
{
	struct address_table  * AddrTab;
	struct function_table * FuncTab;
	int i;

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:MPI_CALLER_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:MPI_CALLER_TYPE];

	if (Address2Info_Labels[A2I_MPI] /*Address2Info_Need_MPI_Labels*/) 
	{
		fprintf(pcf_fd, "%s\n", TYPE_LABEL);
		if (MPI_Caller_Multiple_Levels_Traced) 
		{
			if (MPI_Caller_Labels_Used == NULL) 
			{
				for(i=1; i<=MAX_CALLERS; i++) 
				{
					fprintf(pcf_fd, "0    %d    %s %d\n", CALLER_EV+i, CALLER_LVL_LBL, i);
				}
			}
			else 
			{
				for(i=1; i<=MAX_CALLERS; i++) 
				{
					if (MPI_Caller_Labels_Used[i-1] == TRUE) 
					{
						fprintf(pcf_fd, "0    %d    %s %d\n", CALLER_EV+i, CALLER_LVL_LBL, i);
					}
				}
			}
		}
		else 
		{
			fprintf(pcf_fd, "0    %d    %s\n", CALLER_EV, CALLER_LBL);
		}
		if (Address2Info_Initialized())
		{
			fprintf(pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i=0; i<FuncTab->num_functions; i++) 
			{
				fprintf(pcf_fd, "%d   %s\n", i + 1, FuncTab->function[i]);
			}
			LET_SPACES(pcf_fd);
		}

		fprintf(pcf_fd, "%s\n", TYPE_LABEL);
		if (MPI_Caller_Multiple_Levels_Traced) 
		{
			if (MPI_Caller_Labels_Used == NULL) 
			{
				for(i=1; i<=MAX_CALLERS; i++) 
				{
					fprintf(pcf_fd, "0    %d    %s %d\n", CALLER_LINE_EV+i, CALLER_LINE_LVL_LBL, i);
				}
			}
			else 
			{
				for(i=1; i<=MAX_CALLERS; i++) 
				{
					if (MPI_Caller_Labels_Used[i-1] == TRUE) 
					{
						fprintf(pcf_fd, "0    %d    %s %d\n", CALLER_LINE_EV+i, CALLER_LINE_LVL_LBL, i);
					}
				}
			}
		}
		else 
		{
			fprintf(pcf_fd, "0    %d    %s\n", CALLER_LINE_EV, CALLER_LINE_LBL);
		}
		if (Address2Info_Initialized())
		{
			fprintf(pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < AddrTab->num_addresses; i ++) 
				fprintf(pcf_fd, "%d   %d (%s)\n", 
					i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name);
			LET_SPACES(pcf_fd);
		}
	}
}

void Address2Info_Write_OMP_Labels (FILE * pcf_fd, int eventtype, int eventtype_line, int uniqueid)
{
	struct address_table  * AddrTab;
	struct function_table * FuncTab;
	int i;

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:OUTLINED_OPENMP_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:OUTLINED_OPENMP_TYPE];

	if (Address2Info_Labels[A2I_OMP] /*Address2Info_Need_OMP_Labels*/) 
	{
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", eventtype, "Parallel function");

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < FuncTab->num_functions; i ++)
				fprintf (pcf_fd, "%d   %s\n", i + 1, FuncTab->function[i]);
			LET_SPACES(pcf_fd);
		}

		/* Then dump line-functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", eventtype_line, "Parallel function line");
		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);

			for (i = 0; i < AddrTab->num_addresses; i ++)
				fprintf(pcf_fd, "%d   %d (%s)\n", 
					i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name);
			LET_SPACES(pcf_fd);
		}
	}
}

void Address2Info_Write_CUDA_Labels (FILE * pcf_fd, int uniqueid)
{
	struct address_table  * AddrTab;
	struct function_table * FuncTab;
	int i;

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:CUDAKERNEL_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:CUDAKERNEL_TYPE];

	if (Address2Info_Labels[A2I_CUDA]) 
	{
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", CUDAFUNC_EV, "CUDA kernel");

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < FuncTab->num_functions; i ++)
			{
				char buffer[1024];
				unsigned count = 0;
				char *ptr;

				if ((ptr = strstr (FuncTab->function[i], "__device_stub__Z")) != NULL)
				{
					ptr += strlen ("__device_stub__Z");
					while (ptr[0]>='0' && ptr[0]<='9')
					{
						count = count * 10 + (ptr[0] - '0');
						ptr++;
					}
					/* Add +1 in the demangled name for additional \0 in C */
					snprintf (buffer, MIN(count+1,sizeof(buffer)), ptr);
					fprintf (pcf_fd, "%d   %s\n", i + 1, buffer);
				}
				else
					fprintf (pcf_fd, "%d   %s\n", i + 1, FuncTab->function[i]);
			}
			LET_SPACES(pcf_fd);
		}

		/* Then dump line-functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", CUDAFUNC_LINE_EV, "CUDA kernel source code line");
		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);

			for (i = 0; i < AddrTab->num_addresses; i ++)
				fprintf(pcf_fd, "%d   %d (%s)\n", 
					i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name);
			LET_SPACES(pcf_fd);
		}
	}
}

void Address2Info_Write_UF_Labels (FILE * pcf_fd, int uniqueid)
{
	struct address_table  * AddrTab;
	struct function_table * FuncTab;
	int i;

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:USER_FUNCTION_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:USER_FUNCTION_TYPE];

	if (Address2Info_Labels[A2I_UF] /*Address2Info_Need_UF_Labels*/) 
	{
		/* First dump functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", USRFUNC_EV, "User function");

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < FuncTab->num_functions; i ++)
				fprintf (pcf_fd, "%d   %s\n", i + 1, FuncTab->function[i]);
			LET_SPACES(pcf_fd);
		}

		/* Then dump line-functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", USRFUNC_LINE_EV, "User function line");

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < AddrTab->num_addresses; i ++)
				fprintf(pcf_fd, "%d   %d (%s)\n", 
					i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name);
			LET_SPACES(pcf_fd);
		}
	}
}

void Address2Info_Write_Sample_Labels (FILE * pcf_fd, int uniqueid)
{
	struct address_table  * AddrTab;
	struct function_table * FuncTab;
	int i;

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:SAMPLE_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:SAMPLE_TYPE];

	if (Address2Info_Labels[A2I_SAMPLE] /*Address2Info_Need_Sample_Labels*/) 
	{
		/* First dump functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", SAMPLING_EV, "Sampled functions");
		if (Sample_Caller_Labels_Used != NULL)
			for (i = 1; i <= MAX_CALLERS; i++)
				if (Sample_Caller_Labels_Used[i-1])
					fprintf (pcf_fd, "0    %d    Sampled functions (depth %d)\n", SAMPLING_EV+i, i);

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < FuncTab->num_functions; i ++)
				fprintf (pcf_fd, "%d   %s\n", i + 1, FuncTab->function[i]);
			LET_SPACES(pcf_fd);
		}

		/* Then dump line-functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", SAMPLING_LINE_EV, "Sampled line functions");
		if (Sample_Caller_Labels_Used != NULL)
			for (i = 1; i <= MAX_CALLERS; i++)
				if (Sample_Caller_Labels_Used[i-1])
					fprintf (pcf_fd, "0    %d    Sampled lines functions (depth %d)\n", SAMPLING_LINE_EV+i, i);

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < AddrTab->num_addresses; i ++)
				fprintf(pcf_fd, "%d   %d (%s)\n", 
					i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name);
			LET_SPACES(pcf_fd);
		}
	}
}

#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-aux.h"

void Share_Callers_Usage (void)
{
	int MPI_used[MAX_CALLERS], SAMPLE_used[MAX_CALLERS];
	int A2I_tmp[A2I_LAST];
	int i, res;

	res = MPI_Reduce (Address2Info_Labels, A2I_tmp, A2I_LAST, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK (res, MPI_Reduce, "Sharing information about address<->info labels");
	for (i = 0; i < A2I_LAST; i++)
		Address2Info_Labels[i] = A2I_tmp[i];

	if (MPI_Caller_Labels_Used == NULL)
	{
		MPI_Caller_Labels_Used = malloc(sizeof(int)*MAX_CALLERS);
		if (MPI_Caller_Labels_Used == NULL)
		{
			fprintf (stderr, "mpi2prv: Fatal error! Cannot allocate memory for used MPI Caller labels\n");
			exit (-1);
		}
		for (i = 0; i < MAX_CALLERS; i++)
			MPI_Caller_Labels_Used[i] = FALSE;
	}
	res = MPI_Reduce (MPI_Caller_Labels_Used, MPI_used, MAX_CALLERS, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing information about MPI address<->info");
	for (i = 0; i < MAX_CALLERS; i++)
		MPI_Caller_Labels_Used[i] = MPI_used[i];

	if (Sample_Caller_Labels_Used == NULL)
	{
		Sample_Caller_Labels_Used = malloc(sizeof(int)*MAX_CALLERS);
		if (Sample_Caller_Labels_Used == NULL)
		{
			fprintf (stderr, "mpi2prv: Fatal error! Cannot allocate memory for used sample Caller labels\n");
			exit (-1);
		}
		for (i = 0; i < MAX_CALLERS; i++)
			Sample_Caller_Labels_Used[i] = FALSE;
	}
	res = MPI_Reduce (Sample_Caller_Labels_Used, SAMPLE_used, MAX_CALLERS, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing information about sampling address<->info");
	for (i = 0; i < MAX_CALLERS; i++)
		Sample_Caller_Labels_Used[i] = SAMPLE_used[i];

	res = MPI_Reduce (&MPI_Caller_Multiple_Levels_Traced, &i, 1, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing information about multiple address<->info labels");
	MPI_Caller_Multiple_Levels_Traced = i;
}
#endif /* PARALLEL_MERGE */

#endif /* HAVE_BFD */
