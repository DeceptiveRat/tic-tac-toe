#define _GNU_SOURCE // to expose getopt
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "common.h"
#include "training.h"

extern int errnum;
extern int debug_mode;

int main(int argc, char **argv)
{
	int opt;
	int mode = 1;
	int train_iteration = 5;
	float gamma = 0.5;
	int Q_train_options = 0x01;
	int train_options = 0x01;

	while((opt = getopt(argc, argv, ":htc:g:mard")) != -1)
	{
		switch(opt)
		{
			case 'h':
				printHelp(argv[0]);
				break;
			case 't':
				mode = 1;
				break;
			case 'c':
				train_iteration = atoi(optarg);
				break;
			case 'g':
				gamma = atof(optarg);
			case 'm':
				Q_train_options |= TRAINQTENSOR_USEMAXQ;
				break;
			case 'a':
				Q_train_options |= TRAINQTENSOR_USEAVGQ;
				break;
			case 'r':
				train_options |= SIMULATEGAME_RANDOMSTART;
				break;
			case 'd':
				debug_mode = 1;
				break;
			case '?':
				printf("unknown option: %c\n", optopt);
				break;
		}
	}

	if(mode == 0)
	{
		printHelp(argv[0]);
		return 0;
	}
	else if(mode == 1)
	{
		if(!verifyGamma(gamma))
		{
			printf("Gamma is between 0 and 1!\n");
			return -1;
		}
		if(trainMode(train_iteration, gamma, Q_train_options, train_options) == -1)
		{
			printf("Critical Error!\n");
			printf("Error number: %d\n", errnum);
		}
	}

	return 0;
}
