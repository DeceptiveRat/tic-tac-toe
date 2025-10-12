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

#define TRAINMODE_START 0xff
#define TRAINMODE_RANDOMSTART 0x01

#define TRAINQTENSOR_MODE 0xff
#define TRAINQTENSOR_USEMAXQ 0x01
#define TRAINQTENSOR_USEAVGQ 0x02

int trainMode(const int train_iteration, const float gamma, const int Q_train_options, const int train_options);
int trainQTensor(const int8_t current_state[], int Q_tensor[][27][27][9],
				 const int8_t R_tensor[][27][27][2], const float gamma, const int options);
void printHelp(const char *argv);
int generateRTensor(int8_t current_state[], int8_t R_tensor[][27][27][2]);
bool isGameover(const int8_t current_state[], int8_t player);
bool getRValue(const int8_t current_state[], const int8_t R_tensor[][27][27][2], int8_t result[]);
bool getQMatrix(const int8_t current_state[], const int Q_tensor[][27][27][9], int result[]);
int getHash(const int8_t current_state[]);
int unhash(const int *hash, int8_t return_value[]);
bool setRValue(const int8_t current_state[], int8_t R_tensor[][27][27][2], int8_t R_value[]);
bool setQMatrix(const int8_t current_state[], int Q_tensor[][27][27][9], int Q_matrix[]);
void printMatrix(const int8_t matrix[]);
void resetMatrix(void* matrix, int length);
int chooseRandom(const int8_t current_state[]);
int chooseMaxQValue(const int8_t current_state[], const int Q_tensor[][27][27][9]);
int chooseAverageQValue(const int8_t current_state[], const int Q_tensor[][27][27][9]);
void setErr(int num);
void resetErr();

int errnum = 0;

int main(int argc, char **argv)
{
	int opt;
	int mode = 1;
	int train_iteration = 100;
	float gamma = 0.5;
	int Q_train_options = 0x01;
	int train_options = 0x01;

	while((opt = getopt(argc, argv, ":htc:g:mar")) != -1)
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
				train_options |= TRAINMODE_RANDOMSTART;
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
		trainMode(train_iteration, gamma, Q_train_options, train_options);
	}

	return 0;
}

int trainMode(const int train_iteration, const float gamma, const int Q_train_options, const int train_options)
{
	int8_t R_tensor[27][27][27][2];
	int Q_tensor[27][27][27][9];
	memset(R_tensor, 0, 27 * 27 * 27 * 2);
	memset(Q_tensor, 0, 27 * 27 * 27 * 9*sizeof(int));

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
		int next_move;
		printf("Starting game number[%d]:\n", i);
		if((train_options&TRAINMODE_START)==TRAINMODE_RANDOMSTART)
		{
			int r = rand() % 9;
			current_state[r] = 1;
			printMatrix(current_state);
		}
		else
		{
			// player move
			next_move = trainQTensor(current_state, Q_tensor, R_tensor, gamma, Q_train_options);
			if(errnum)
				return -1;
			current_state[next_move] = PLAYER;
			if(isGameover(current_state, PLAYER))
				break;
			printMatrix(current_state);
		}
		while(1)
		{
			// opponent next move
			next_move = chooseRandom(current_state);
			if(errnum == 1)
			{
				printf("Tie!\n");
				resetErr();
				break;
			}
			else if(errnum == 2)
			{
				return -1;
			}

			current_state[next_move] = OPPONENT;
			printMatrix(current_state);
			if(isGameover(current_state, OPPONENT))
			{
				printf("Opponent Wins!\n");
				break;
			}

			// player move
			next_move = trainQTensor(current_state, Q_tensor, R_tensor, gamma, Q_train_options);
			if(errnum)
				return -1;
			current_state[next_move] = PLAYER;
			printMatrix(current_state);
			if(isGameover(current_state, PLAYER))
			{
				printf("Player Wins!\n");
				break;
			}
		}
		resetMatrix(current_state, sizeof(int8_t));
	}

	return 0;
}

int trainQTensor(const int8_t current_state[], int Q_tensor[][27][27][9],
				 const int8_t R_tensor[][27][27][2], const float gamma, const int options)
{
	int next_move = chooseRandom(current_state);
	int8_t next_state[9];
	memcpy(next_state, current_state, 9*sizeof(int8_t));
	next_state[next_move] = PLAYER;

	int8_t R_value[2];
	getRValue(current_state, R_tensor, R_value);
	int next_move_reward = (R_value[0] == next_move) ? (R_value[1]) : (0);

	int Q_matrix[9];
	getQMatrix(current_state, Q_tensor, Q_matrix);
	if(errnum)
		return -1;
	int Q_current = Q_matrix[next_move];

	int updated_Q_value;
	if((options & TRAINQTENSOR_MODE) == TRAINQTENSOR_USEMAXQ)
	{
		int Q_max = chooseMaxQValue(next_state, Q_tensor);
		if(errnum)
			return -1;

		updated_Q_value = (1-gamma)*Q_current + next_move_reward + gamma * Q_max;
	}
	else if((options & TRAINQTENSOR_MODE) == TRAINQTENSOR_USEAVGQ)
	{
		int Q_avg = chooseAverageQValue(next_state, Q_tensor);
		if(errnum)
			return -1;

		updated_Q_value = (1-gamma)*Q_current + next_move_reward + gamma * Q_avg;
	}

	Q_matrix[next_move] = updated_Q_value;
	setQMatrix(current_state, Q_tensor, Q_matrix);

	return next_move;
}

int chooseMaxQValue(const int8_t current_state[], const int Q_tensor[][27][27][9])
{
	int Q_matrix[9];
	resetMatrix(Q_matrix, sizeof(int));
	if(!getQMatrix(current_state, Q_tensor, Q_matrix))
		errnum = 1;

	int max = -1;
	for(int i = 0; i < 9; i++)
	{
		// not available spot
		if(current_state[i] != 0)
			continue;
		if(Q_matrix[i] > max)
			max = Q_matrix[i];
	}

	return max;
}

void setErr(int num) { if(num==0) return; errnum = num;}
void resetErr() {errnum = 0;};

int chooseAverageQValue(const int8_t current_state[], const int Q_tensor[][27][27][9])
{
	int Q_matrix[9];
	resetMatrix(Q_matrix, sizeof(int));
	if(!getQMatrix(current_state, Q_tensor, Q_matrix))
	{
		setErr(1);
		return -1;
	}

	int sum = 0;
	for(int i = 0; i < 9; i++)
		sum += Q_matrix[i];

	int empty_count = 0;
	for(int i = 0; i < 9; i++)
	{
		if(current_state[i] != 0)
			empty_count++;
	}

	return sum / empty_count;
}

bool verifyQMatrix(const int8_t current_state, const int Q_matrix)
{
	// if Q matrix is 0 in all non empty spots
	return true;
	// else
	return false;
}

int chooseRandom(const int8_t current_state[])
{
	int empty_count = 0;
	// find empty spots
	for(int i = 0; i < 9; i++)
	{
		if(current_state[i] == 0)
			empty_count++;
	}

	// no where left to place
	if(empty_count == 0)
	{
		setErr(1);
		return -1;
	}

	// choose random empty spot
	empty_count = (rand() % empty_count) + 1;
	for(int i = 0; i < 9; i++)
	{
		if(current_state[i] == 0)
			empty_count--;
		if(empty_count == 0)
			return i;
	}

	// not supposed to happen
	setErr(2);
	return -1;
}

void printHelp(const char *argv)
{
	printf("Usage: %s\n", argv);
	printf("options:");
	printf("-h: for this help message\n");
	printf("-t: to train model (default)\n");
	printf("-c <int>: to set train iteration count (100 by default)\n");
	printf("-g <float>: set gamma (0.5 by default)\n");
	printf("-m: [Q train mode] train Q matrix with max values (default)(do not use with other Q train modes!)\n");
	printf("-a: [Q train mode] train Q matrix with avg values (do not use with other Q train modes!)\n");
	printf("-r: [train mode] Player starts at random location (default)(do not use with other train modes!)\n");
}

int generateRTensor(int8_t current_state[], int8_t R_tensor[][27][27][2])
{
	int8_t R_value[2];
	memset(R_value, 0, 2);
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
			R_value[0] = i + 1;
			R_value[1] = reward;
			if(!setRValue(current_state, R_tensor, R_value))
			{
				setErr(1);
				return -1;
			}

			memset(R_value, 0, 2);
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

bool getRValue(const int8_t current_state[], const int8_t R_tensor[][27][27][2], int8_t result[])
{
	int hash = getHash(current_state);
	if((hash & 0xff000000) != 0x00)
	{
		setErr(1);
		return false;
	}

	memcpy(result, R_tensor[hash >> 24][hash >> 16][hash >> 8], 2);
	return true;
}

bool getQMatrix(const int8_t current_state[], const int Q_tensor[][27][27][9], int result[])
{
	int hash = getHash(current_state);
	if((hash & 0xff000000) != 0x00)
	{
		setErr(1);
		return false;
	}

	memcpy(result, Q_tensor[hash >> 24][hash >> 16][hash >> 8], 9*sizeof(int));
	return true;
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

bool setRValue(const int8_t current_state[], int8_t R_tensor[][27][27][2], int8_t R_value[])
{
	int hash = getHash(current_state);
	if((hash & 0xff000000) != 0x00)
	{
		setErr(1);
		return false;
	}
	memcpy(R_tensor[(hash >> 16) & 0xff][(hash >> 8) & 0xff][(hash) & 0xff], R_value, 2);
	return true;
}

bool setQMatrix(const int8_t current_state[], int Q_tensor[][27][27][9], int Q_matrix[])
{
	int hash = getHash(current_state);
	if((hash & 0xff000000) != 0x00)
	{
		setErr(1);
		return false;
	}
	memcpy(Q_tensor[(hash >> 16) & 0xff][(hash >> 8) & 0xff][(hash) & 0xff], Q_matrix, 9*sizeof(int));
	return true;
}

void printMatrix(const int8_t matrix[])
{
	printf("printing matrix...\n");
	printf("1: %-3d %8d %8d\n2: %-3d %8d %8d\n3: %-3d %8d %8d\n", matrix[0], matrix[1], matrix[2],
		   matrix[3], matrix[4], matrix[5], matrix[6], matrix[7], matrix[8]);
}

void resetMatrix(void* matrix, int length) { memset(matrix, 0, 9*length); }

int chooseMove(const int8_t current_state[], const int8_t R_tensor[][27][27][2])
{
	int8_t R_value[2];
	getRValue(current_state, R_tensor, R_value);

	if(R_value[1] <= 0)
	{
	}
}
