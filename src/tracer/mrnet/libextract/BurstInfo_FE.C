#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include "BurstInfo_FE.h"

static void BurstInfo_Dump (BurstInfo_t *bi, int fd)
{
    write (fd, &bi->TaskID, sizeof(int));
    write (fd, &bi->ThreadID, sizeof(int));

    write (fd, &bi->num_Bursts, sizeof(int));
    write (fd, &bi->num_HWCperBurst, sizeof(int));

    write (fd, bi->Timestamp, bi->num_Bursts * sizeof(long long));
    write (fd, bi->Durations, bi->num_Bursts * sizeof(long long));
    write (fd, bi->HWCValues, bi->num_Bursts * bi->num_HWCperBurst * sizeof(long long));
    write (fd, bi->HWCSet, bi->num_Bursts * sizeof(int));
}   

void BurstInfo_DumpArray (BurstInfo_t **bi_list, int count, const char *out_suffix, std::map< int, std::vector<int> > & HWC_Sets_Ids)
{
    int i, fd;
    char DumpFile[256];

    snprintf(DumpFile, sizeof(DumpFile), "DATA_%s.bbi", out_suffix);

    if ((fd = open(DumpFile, O_WRONLY | O_CREAT | O_TRUNC, 0644)) == -1)
    {
        perror("open");
        exit(1);
    }

	std::map< int, std::vector<int> >::iterator it;
	int num_sets = HWC_Sets_Ids.size();
	write(fd, &num_sets, sizeof(int));
	for (it=HWC_Sets_Ids.begin(); it!=HWC_Sets_Ids.end(); it++)
	{
		int set = it->first;
		std::vector<int> v = it->second;
		int hwcs_in_set = v.size();

		write(fd, &set, sizeof(int));
		write(fd, &hwcs_in_set, sizeof(int));
		write(fd, &v[0], sizeof(int) * v.size());
	}

    write (fd, &count, sizeof(int));

    for (i=0; i<count; i++)
    {
        BurstInfo_Dump ( bi_list[i], fd );
    }
}

static BurstInfo_t * BurstInfo_Load (int fd)
{
    int TaskID, ThreadID, num_Bursts, num_HWCperBurst;
    BurstInfo_t *bi = NULL;

    read (fd, &TaskID, sizeof(int));
    read (fd, &ThreadID, sizeof(int));
    read (fd, &num_Bursts, sizeof(int));
    read (fd, &num_HWCperBurst, sizeof(int));

    bi = new_BurstInfo (TaskID, ThreadID, num_Bursts, num_HWCperBurst);
    bi->num_Bursts = num_Bursts;

    if (bi != NULL)
    {
        read (fd, bi->Timestamp, num_Bursts * sizeof(long long));
        read (fd, bi->Durations, num_Bursts * sizeof(long long));
        read (fd, bi->HWCValues, num_Bursts * num_HWCperBurst * sizeof(long long));
        read (fd, bi->HWCSet, num_Bursts * sizeof(int));
    }
    return bi;
}

int BurstInfo_LoadArray (char *FileBBI, BurstInfo_t ***bi_io, std::map< int, std::vector<int> > * HWC_Sets_Ids)
{
    int i, j, fd, count = 0;
	BurstInfo_t **bi_list = NULL;
    std::map< int, std::vector<int> > m;
	int num_sets, set, hwcs_in_set, *labels;

    if ((fd = open(FileBBI, O_RDONLY)) == -1)
    {
        perror("open");
        exit(1);
    }

	read (fd, &num_sets, sizeof(int));
	for (i=0; i<num_sets; i++)
	{
		read (fd, &set, sizeof(int));
		read (fd, &hwcs_in_set, sizeof(int));
		labels = (int *)malloc(sizeof(int) * hwcs_in_set);
		read (fd, labels, sizeof(int) * hwcs_in_set);
		for (j=0; j<hwcs_in_set; j++)
		{
			m[set].push_back(labels[j]);
		}
		free(labels);
	}
	*HWC_Sets_Ids = m;

    read (fd, &count, sizeof(int));
	bi_list = (BurstInfo_t **)malloc(count * sizeof(BurstInfo_t *));
    for (i=0; i<count; i++)
    {
		bi_list[i] = BurstInfo_Load (fd);
    }
	*bi_io = bi_list;
    return count;
}

