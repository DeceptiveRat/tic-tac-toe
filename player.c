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
	int8_t R_tensor[19683][2];
	int Q_tensor[19683][9];
	memset(R_tensor, 0xff, 19683 * 2 * sizeof(int8_t));
	memset(Q_tensor, 0, 19683 * 9 * sizeof(int));

	int opt;
	int mode = 0;
	int train_iteration = 5;
	float gamma = 0.5;
	int Q_train_options = 0;
	Q_train_options &= TRAINQTENSOR_USEMAXQ;
	int simulate_game_options = 0;
	simulate_game_options &= SIMULATEGAME_RANDOMSTART;
	srand(time(NULL));
	bool save_tensors = false;
	bool load_tensors = false;

	while((opt = getopt(argc, argv, ":htc:g:mardslop")) != -1)
	{
		switch(opt)
		{
			case 'h':
				printHelp(argv[0]);
				exit(-1);
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
				Q_train_options &= !TRAINQTENSOR_MODE;
				Q_train_options |= TRAINQTENSOR_USEMAXQ;
				break;
			case 'a':
				Q_train_options &= !TRAINQTENSOR_MODE;
				Q_train_options |= TRAINQTENSOR_USEAVGQ;
				break;
			case 'r':
				simulate_game_options &= !SIMULATEGAME_OPTIONS;
				simulate_game_options |= SIMULATEGAME_RANDOMSTART;
				break;
			case 'd':
				debug_mode = 1;
				break;
			case 's':
				save_tensors = true;
				break;
			case 'l':
				load_tensors = true;
				break;
			case 'o':
				simulate_game_options |= SIMULATEGAME_PLAYERP2;
				break;
			case 'p':
				simulate_game_options |= SIMULATEGAME_PLAYERP1;
				break;
			case '?':
				printf("unknown option: %c\n", optopt);
				break;
		}
	}

	if(mode == 1)
	{
		if(!verifyGamma(gamma))
		{
			printf("Gamma is between 0 and 1!\n");
			return -1;
		}
		if(trainMode(Q_tensor, R_tensor, train_iteration, gamma, Q_train_options, simulate_game_options) == -1)
		detectError();
		
		if(save_tensors)
		{
			saveTensors(Q_tensor, QFILE, R_tensor, RFILE);
			detectError();
		}
	}

	DEBUG_EXEC(printQTensor(Q_tensor));

	if(load_tensors)
	{
		loadTensors(Q_tensor, QFILE, R_tensor, RFILE);
		detectError();
	}

	int results[3] = {0};
	int game_count=100000;
	for(int i = 0;i<game_count;i++)
	{
		int result;
		result = playGame(Q_tensor);
		detectError();
		if(result == P1)
			results[0]++;
		else if(result == TIE)
			results[1]++;
		else
			results[2]++;
	}

	printf("P1 win rate: %.3f\n", (float)results[0]/game_count);
	printf("Tie rate: %.3f\n", (float)results[1]/game_count);
	printf("P2 win rate: %.3f\n", (float)results[2]/game_count);

	return 0;
}
