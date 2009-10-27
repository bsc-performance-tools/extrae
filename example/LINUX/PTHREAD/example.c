#include <stdio.h>
#include <pthread.h>

#define MAX_THREADS 32

void *routine1 (void *parameters)
{
        SEQtrace_event (1, 1);
        printf ("routine1 : (thread=%08x, param %p)\n", pthread_self(), parameters);
        SEQtrace_event (1, 0);
}

void *routine2 (void *parameters)
{
        SEQtrace_event (2, 1);
        printf ("routine 2 : (thread=%08x, param %p)\n", pthread_self(), parameters);
        SEQtrace_event (2, 0);
}


int main (int argc, char *argv[])
{
        pthread_t t[MAX_THREADS];
        int i;

        MPItrace_init ();

        for (i = 0; i < MAX_THREADS; i++)
                pthread_create (&t[i], NULL, routine1, (void*) ((long) i));
        for (i = 0; i < MAX_THREADS; i++)
                pthread_join (t[i], NULL);

        sleep (1);

        for (i = 0; i < MAX_THREADS; i++)
                pthread_create (&t[i], NULL, routine2, NULL);
        for (i = 0; i < MAX_THREADS; i++)
                pthread_join (t[i], NULL);

        MPItrace_fini();
}


