#define _GNU_SOURCE // to expose getopt
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define PLAYER 1
#define OPPONENT -1

int trainMatrix(const int train_iteration);
void printHelp(const char *argv);
int generateRTensor(int8_t current_state[], int8_t R_tensor[][27][27][9]);
bool isGameover(const int8_t current_state[], int8_t player);
int getRMatrix(const int8_t current_state[], const int8_t R_tensor[][27][27][9], int8_t result[]);
int getHash(const int8_t current_state[]);
int unhash(const int *hash, int8_t return_value[]);
bool setRMatrix(const int8_t current_state[], int8_t R_tensor[][27][27][9], int8_t R_matrix[]);
void printMatrix(const int8_t matrix[]);
void resetMatrix(int8_t matrix[]);

int main(int argc, char **argv)
{
	int opt;
	int mode = 1;
	int train_iteration = 5;

	while((opt = getopt(argc, argv, ":htc:")) != -1)
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
			case '?':
				printf("unknown option: %c\n", optopt);
				break;
		}
	}

	if(mode == 0)
	{
		printHelp(argv[0]);
		exit(-1);
	}
	else if(mode == 1)
	{
		trainMatrix(train_iteration);
	}

	return 0;
}

int trainMatrix(const int train_iteration)
{
	int8_t R_tensor[27][27][27][9];
	int8_t Q_tensor[27][27][27][9];
	memset(R_tensor, 0, 27 * 27 * 27 * 9);
	memset(Q_tensor, 0, 27 * 27 * 27 * 9);

	int8_t current_state[9] = {0};
	// create R tensor
	if(generateRTensor(current_state, R_tensor) == -1)
	{
		printf("error generating R tensor\n");
		exit(-1);
	}

	// training session
	srand(time(NULL));
	for(int i = 0; i < train_iteration; i++)
	{
		int r = rand() % 9;
		current_state[r] = 1;
		while(!isGameover(current_state, PLAYER) && !isGameover(current_state, OPPONENT))
		{
			int8_t choice_matrix[8];
		}
		resetMatrix(current_state);
	}

	return 0;
}

void printHelp(const char *argv)
{
	printf("Usage: %s\n", argv);
	printf("options:");
	printf("-h: for this help message\n");
	printf("-t: to train model (default)\n");
	printf("-c: to set train iteration count (100 by default)\n");
}

int generateRTensor(int8_t current_state[], int8_t R_tensor[][27][27][9])
{
	int8_t R_matrix[9];
	memcpy(R_matrix, current_state, 9);
	for(int i = 0; i < 9; i++)
	{
		if(current_state[i] != 0)
			continue;
		else
		{
			int reward = 0;
			current_state[i] = PLAYER;
			if(isGameover(current_state, PLAYER))
				reward = 100;
			else
			{
				int branch_count = 0;
				for(int j = 0; j < 9; j++)
				{
					if(current_state[j] != 0)
						continue;
					else
					{
						branch_count++;
						current_state[j] = OPPONENT;
						if(isGameover(current_state, OPPONENT))
							reward -= 100;
						else
							generateRTensor(current_state, R_tensor);
						current_state[j] = 0;
					}
				}
				if(reward != 0)
					reward /= branch_count;
			}

			current_state[i] = 0;
			R_matrix[i] = reward;
			if(!setRMatrix(current_state, R_tensor, R_matrix))
				return -1;

			R_matrix[i] = 0;
		}
	}

	return 0;
}

bool isGameover(const int8_t current_state[], int8_t player)
{
	int end_conditions[8] = {0x1c0, 0x38, 0x7, 0x124, 0x92, 0x49, 0x111, 0x54};
	int current_state_bits = 0;
	for(int i = 0; i < 9; i++)
	{
		if(current_state[i] == player)
			current_state_bits += 1 << (8 - i);
	}

	for(int8_t i = 0; i < 8; i++)
	{
		if((current_state_bits & end_conditions[i]) == end_conditions[i])
			return true;
	}

	return false;
}

int getRMatrix(const int8_t current_state[], const int8_t R_tensor[][27][27][9], int8_t result[])
{
	int hash = getHash(current_state);
	if((hash & 0xff000000) != 0x00)
		return -1;

	memcpy(result, R_tensor[hash >> 24][hash >> 16][hash >> 8], 9);
	return 0;
}

int getHash(const int8_t current_state[])
{
	int8_t indices[4] = {0};

	for(int i = 0; i < 3; i++)
	{
		for(int j = 0; j < 3; j++)
		{
			int8_t byte = current_state[3 * i + j];
			byte += 1;

			if(j == 0)
				indices[i] += 3 * 3 * byte;
			else if(j == 1)
				indices[i] += 3 * byte;
			else
				indices[i] += byte;
		}
	}

	return *(int *)(indices);
}

int unhash(const int *hash, int8_t return_value[])
{
	int8_t *indices;
	indices = (int8_t *)(hash);
	for(int i = 0; i < 3; i++)
	{
		int8_t byte;
		byte = indices[i] / (3 * 3);
		byte--;
		return_value[i * 3] = byte;

		byte = (indices[i] / 3) % 3;
		byte--;
		return_value[i * 3 + 1] = byte;

		byte = indices[i] % 3;
		byte--;
		return_value[i * 3 + 2] = byte;
	}

	return 0;
}

bool setRMatrix(const int8_t current_state[], int8_t R_tensor[][27][27][9], int8_t R_matrix[])
{
	int hash = getHash(current_state);
	if((hash & 0xff000000) != 0x00)
		return false;
	memcpy(R_tensor[(hash >> 16) & 0xff][(hash >> 8) & 0xff][(hash) & 0xff], R_matrix, 9);
	return true;
}

void printMatrix(const int8_t matrix[])
{
	printf("printing matrix...\n");
	printf("1: %-3d %8d %8d\n2: %-3d %8d %8d\n3: %-3d %8d %8d\n", matrix[0], matrix[1], matrix[2],
		   matrix[3], matrix[4], matrix[5], matrix[6], matrix[7], matrix[8]);
}

void resetMatrix(int8_t matrix[]) { memset(matrix, 0, 9); }

int chooseMove(const int8_t current_state[], const int8_t R_tensor[][27][27][9])
{
	int8_t R_matrix[9];
	getRMatrix(current_state, R_tensor, R_matrix);

	int max_index = 0;
	int zero_count = 0;
	for(int i = 1; i < 9; i++)
	{
		if(R_matrix[i] > R_matrix[max_index])
			max_index = i;
	}

	if(R_matrix[max_index] == 0)
	{
		int 
	}
}
