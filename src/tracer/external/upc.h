#ifndef _UPC_H_INCLUDED_
#define _UPC_H_INCLUDED_

/* These routines are provided by UPC. Contact Montserrat Farreras
   for further info */

/* Threads in current processor */
unsigned GetNumUPCthreads(void);
unsigned GetUPCthreadID(void);

/* Processor in current application */
unsigned GetNumUPCprocs(void);
unsigned GetUPCprocID(void);

#endif /* _UPC_H_INCLUDED_ */
