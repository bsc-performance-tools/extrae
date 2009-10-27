#ifndef gm_zone_types_h		/* -*-c-*- */
#define gm_zone_types_h

#include "gm_types.h"

typedef struct gm_zone_area
{
  struct gm_zone_area   *next;
  struct gm_zone_area   *prev;
} gm_zone_area_t;

typedef struct gm_zone
{
  gm_zone_area_t	*base;
  gm_u32_t		length;
  gm_zone_area_t	free_list[32];
  gm_u32_t		*boundary_bits;
  gm_u32_t		*reserved_bits;
} gm_zone_t;

#endif
