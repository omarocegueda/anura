#include "tictoc.h"
#include "time.h"
#include <stdio.h>
time_t tictoc_start_time=0;
 
void tic(void){
	tictoc_start_time=clock();
}
void toc(void){
	time_t current_time=clock();
	double elapsed = (double) (current_time - tictoc_start_time) / CLOCKS_PER_SEC; 
	printf("Elapsed time: %lf s.\n", elapsed);
}