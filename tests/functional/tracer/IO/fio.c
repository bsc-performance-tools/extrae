#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>

int main(int argc, char **argv)
{
	FILE *fp;
	int sz;
	char *str = "Hello world\n";

	if ((fp = fopen("tmp.file", "w+")) == NULL)
	{
		perror("fopen test");
	}

	if ((sz = fwrite(str, 1, strlen(str), fp)) < 0)
	{
		perror("fwrite test");
	}

	str = (char *)calloc(128, sizeof(char));

	if ((sz = fread(str, 32, 1, fp)) < 0)
	{
		perror("fread test");
	}

	free(str);

	if (fclose(fp) == EOF)
	{
		perror("fclose test");
	}

	return 0;
}
