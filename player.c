#define _GNU_SOURCE // to expose getopt
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>

#define PLAYER 1
#define OPPONENT -1

void printHelp(const char *argv);
int getPossibleOutcomes(int8_t current_state[], int8_t R_tensor[][27][27][9]);
bool isGameover(const int8_t current_state[], int8_t player);
int8_t* getRMatrix(const int8_t current_state[], int8_t R_tensor[][27][27][9]);
int getHash(const int8_t current_state[]);
int unhash(const int* hash, int8_t return_value[]);
bool setRMatrix(const int8_t current_state[], int8_t R_tensor[][27][27][9], int8_t R_matrix[]);

int main(int argc, char **argv)
{
	int opt;
	int mode = 0;
	int8_t R_tensor[27][27][27][9];
	int8_t Q_tensor[27][27][27][9];
	memset(R_tensor, 0, 27*27*27*9);
	memset(Q_tensor, 0, 27*27*27*9);

	int8_t test[9] = {0, -1,-1, 0,-1, -1, -1, -1, -1};
	int8_t result[9];
	int hash = getHash(test);
	unhash(&hash, result);
	printf("%d\n", memcmp(test, result, 9));

	while((opt = getopt(argc, argv, ":ht")) != -1)
	{
		switch(opt)
		{
			case 'h':
				printHelp(argv[0]);
			case 't':
				mode = 1;
		}
	}

	if(mode == 0)
	{
		printHelp(argv[0]);
		exit(-1);
	}
	else if(mode == 1)
	{
		int8_t current_state[9] = {0};
		if(getPossibleOutcomes(current_state, R_tensor) == -1)
		{
			printf("error getting outcomes\n");
			exit(-1);
		}

		for(int i = 0;i<27;i++)
		{
			for(int j = 0;j<27;j++)
			{
				for(int k =0;k<27;k++)
				{
					int8_t empty[9] = {0};
					if(memcmp(empty, R_tensor[i][j][k], 9) != 0)
					{
						printf("==============================================================\n");
						printf("i: %2d, j: %2d, k: %2d\n", i, j, k);
						int8_t temp[9];
						int hash = (i)|(j<<8)|(k<<16);
						printf("board:\n");
						unhash(&hash, temp);
						printf("%-3d %8d %8d\n%-3d %8d %8d\n%-3d %8d %8d\n", temp[0], temp[1], temp[2], temp[3], temp[4], temp[5], temp[6], temp[7], temp[8]);
						printf("R matrix:\n");
						memcpy(temp, R_tensor[i][j][k], 9);
						printf("%-3d %8d %8d\n%-3d %8d %8d\n%-3d %8d %8d\n", temp[0], temp[1], temp[2], temp[3], temp[4], temp[5], temp[6], temp[7], temp[8]);
					}
				}
			}
		}
	}

	return 0;
}

void printHelp(const char *argv)
{
	printf("Usage: %s\n", argv);
	printf("options:");
	printf("-h: for this help message\n");
	printf("-t: to train model\n");
}

int getPossibleOutcomes(int8_t current_state[], int8_t R_tensor[][27][27][9])
{
	int8_t R_matrix[9];
	memcpy(R_matrix, current_state, 9);
	for(int i = 0;i<9;i++)
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
				for(int j = 0;j<9;j++)
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
							getPossibleOutcomes(current_state, R_tensor);
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
	for(int i = 0;i<9;i++)
	{
		if(current_state[i] == player)
			current_state_bits += 1<<(8-i);
	}

	for(int8_t i = 0;i<8;i++)
	{
		if((current_state_bits & end_conditions[i]) == end_conditions[i])
			return true;
	}

	return false;
}

int8_t* getRMatrix(const int8_t current_state[], int8_t R_tensor[][27][27][9])
{
	int hash = getHash(current_state);
	if((hash&0xff000000) != 0x00)
		return NULL;
	return R_tensor[hash>>24][hash>>16][hash>>8];
}

int getHash(const int8_t current_state[])
{
	int8_t indices[4] = {0};

	for(int i = 0;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
			int8_t byte = current_state[3*i+j];
			byte+=1;

			if(j==0)
				indices[i] += 3*3*byte;
			else if(j==1)
				indices[i] += 3*byte;
			else
				indices[i] += byte;
		}
	}

	return *(int*)(indices);
}

int unhash(const int* hash, int8_t return_value[])
{
	int8_t *indices;
	indices = (int8_t*)(hash);
	for(int i = 0;i<3;i++)
	{
		int8_t byte;
		byte = indices[i]/(3*3);
		byte--;
		return_value[(2-i)*3] = byte;

		byte = (indices[i]/3)%3;
		byte--;
		return_value[(2-i)*3 + 1] = byte;

		byte = indices[i]%3;
		byte--;
		return_value[(2-i)*3 + 2] = byte;
	}

	return 0;
}

bool setRMatrix(const int8_t current_state[], int8_t R_tensor[][27][27][9], int8_t R_matrix[])
{
	int hash = getHash(current_state);
	if((hash&0xff000000) != 0x00)
		return false;
	memcpy(R_tensor[(hash>>16)&0xff][(hash>>8)&0xff][(hash)&0xff],R_matrix, 9);
	return true;
}
