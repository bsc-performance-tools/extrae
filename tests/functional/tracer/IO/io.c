#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>

int main(int argc, char **argv)
{
	int fd, sz;
	char *str = "Hello world\n";

	if ((fd = open("tmp.file", O_RDWR | O_CREAT, 0600)) < 0)
	{
		perror("Open test");
	}

	if ((sz = write(fd, str, strlen(str))) < 0)
	{
		perror("Write test");
	}

	str = (char *)calloc(128, sizeof(char));

	if ((sz = read(fd, str, 32)) < 0)
	{
		perror("Read test");
	}

	free(str);

	if (close(fd) < 0)
	{
		perror("Close test");
	}

	return 0;
}
