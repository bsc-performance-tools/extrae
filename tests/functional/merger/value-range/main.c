#include <extrae.h>
#include <limits.h>
#include <stdio.h>
int main(){
        Extrae_init();
        extrae_value_t value = ULLONG_MAX -1; // ULLONG_MAX -1 = 18446744073709551614
        extrae_type_t type=100;
        printf("Extrae Value: %llu, %u", value, type);
        Extrae_event(type,value);
        extrae_value_t values[2]={0,value};
        char * descriptions[2] = {"zero","max_value-1"};
        unsigned nValues=2;
        Extrae_define_event_type (&type,"ownEventType",&nValues, values, descriptions);
        Extrae_fini();
}