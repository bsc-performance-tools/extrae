
/*
C: The Complete Reference, 4th Ed. (Paperback)
by Herbert Schildt

ISBN: 0072121246
Publisher: McGraw-Hill Osborne Media; 4 edition (April 26, 2000)
*/

/*
Adapted to sort Extrae event records by German Llort (September 9, 2013)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qs_disk.h"
#include "record.h"

void qs_disk(FILE *fp, int left, int right);
void swap_all_fields(FILE *fp, long i, long j);
UINT64 get_timestamp(FILE *fp, long rec);

/* A Quicksort for files. */
void quicksort_disk(char *mpit)
{
  FILE *fp = NULL;
  long mpit_size = 0;
  int num_records = 0;

  if((fp=fopen(mpit, "r+"))==NULL) {
    fprintf(stderr, "Cannot open mpit file '%s'.\n", mpit);
    exit(1);
  }

  fseek(fp, 0L, SEEK_END);
  mpit_size = ftell(fp);
  fseek(fp, 0L, SEEK_SET);
  num_records = (int)(mpit_size / sizeof(event_t));

  qs_disk(fp, 0, num_records-1);
}

void qs_disk(FILE *fp, int left, int right)
{
  long int i, j;
  UINT64 x;

  i = left; j = right;

  x = get_timestamp(fp, (long)(i+j)/2); /* get the middle timestamp */

  do {
    while((get_timestamp(fp,i) < x) && (i < right)) i++;
    while((get_timestamp(fp,j) > x) && (j > left)) j--;

    if(i <= j) {
      swap_all_fields(fp, i, j);
      i++; j--;
    }
  } while(i <= j);

  if(left < j) qs_disk(fp, left, (int) j);
  if(i < right) qs_disk(fp, (int) i, right);
}

void swap_all_fields(FILE *fp, long i, long j)
{
  char a[sizeof(event_t)], b[sizeof(event_t)];

  /* first read in record i and j */
  fseek(fp, sizeof(event_t)*i, SEEK_SET);
  fread(a, sizeof(event_t), 1, fp);

  fseek(fp, sizeof(event_t)*j, SEEK_SET);
  fread(b, sizeof(event_t), 1, fp);

  /* then write them back in opposite slots */
  fseek(fp, sizeof(event_t)*j, SEEK_SET);
  fwrite(a, sizeof(event_t), 1, fp);
  fseek(fp, sizeof(event_t)*i, SEEK_SET);
  fwrite(b, sizeof(event_t), 1, fp);
}

/* Return a pointer to the zip code */
UINT64 get_timestamp(FILE *fp, long rec)
{
  event_t evt;

  fseek(fp, rec*sizeof(event_t), SEEK_SET);
  fread(&evt, sizeof(event_t), 1, fp);

  return Get_EvTime( &evt );
}

