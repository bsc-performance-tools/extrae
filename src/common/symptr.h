#pragma once

#define _GNU_SOURCE
#include <dlfcn.h>

#include "config.h"

/**
 * XTR_FIND_SYMBOL
 *
 * Initialization routine for the dynamic libraries tracing module. Performs a
 * discovery of the address of the real implementation of the calls through
 * dlsym. The initialization is deferred until any of the instrumented symbols
 * is used for the first time.
 */
#if defined(PIC) /* Only available for .so libraries */
#if !defined(DEBUG)
# define XTR_FIND_SYMBOL(func)                                      \
  ({                                                                \
     dlsym (RTLD_NEXT, func);                                       \
  })
#else /* DEBUG */
# include <stdio.h>

# define XTR_FIND_SYMBOL(func)                                      \
  ({                                                                \
     void *ptr = dlsym (RTLD_NEXT, func);                           \
     fprintf (stderr, PACKAGE_NAME": [DEBUG] XTR_FIND_SYMBOL: "     \
                      "Getting pointer to real symbol '%s' (%p)\n", \
                      func, ptr);                                   \
     ptr;                                                           \
  })
#endif /* DEBUG */
#else /* PIC */
# define XTR_FIND_SYMBOL(func)                                      \
  ({                                                                \
     fprintf (stderr, PACKAGE_NAME": Warning! %s instrumenation "   \
                      "requieres linking with shared library!\n",   \
                      func );                                       \
  })
#endif /* PIC */


#define XTR_FIND_SYMBOL_OR_DIE(func)                                \
({                                                                  \
  void *ptr = XTR_FIND_SYMBOL(func);                                \
  if (ptr == NULL)                                                  \
  {                                                                 \
    fprintf(stderr, PACKAGE_NAME": XTR_FIND_SYMBOL: "               \
		    "Failed to find symbol '%s'\n", func);          \
    exit(EXIT_FAILURE);                                             \
  }                                                                 \
  ptr;                                                              \
})
