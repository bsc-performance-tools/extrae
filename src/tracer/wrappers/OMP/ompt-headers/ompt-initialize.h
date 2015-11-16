#ifndef OMPT_INITIALIZE_H_INCLUDED
#define OMPT_INITIALIZE_H_INCLUDED

typedef void (*ompt_initialize_fn_t) (
  ompt_function_lookup_t lookup,
  const char *runtime_version,
  unsigned int ompt_version
);

#endif /* OMPT_INITIALIZE_H_INCLUDED */
