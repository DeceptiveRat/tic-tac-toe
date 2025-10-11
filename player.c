#define _GNU_SOURCE // to expose getopt
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#define PLAYER 1
#define OPPONENT -1

void printHelp(const char *argv);
int getPossibleOutcomes(int8_t current_state[], int8_t* R_tensor[][27][27]);
bool isGameover(const int8_t current_state[], int8_t player);
int8_t* getRMatrix(const int8_t current_state[], int8_t* R_tensor[][27][27]);
int getHash(const int8_t current_state[]);
bool setRMatrix(const int8_t current_state[], int8_t* R_tensor[][27][27], int8_t R_matrix[]);

int main(int argc, char **argv)
{
	int opt;
	int mode = 0;
	int8_t *R_tensor[27][27][27];
	int8_t *Q_tensor[27][27][27];
	memset(R_tensor, 0, 27*27*27);
	memset(Q_tensor, 0, 27*27*27);

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
		getPossibleOutcomes(current_state, R_tensor);
		for(int i = 0;i<27;i++)
		{
			for(int j = 0;j<27;j++)
			{
				for(int k =0;k<27;k++)
				{
					if(R_tensor[i][j][k] != NULL)
					{
						printf("==============================================================\n");
						int8_t temp[9];
						memcpy(temp, R_tensor[i][j][k], 9);
						printf("%d %d %d\n%d %d %d\n%d %d %d\n", temp[0], temp[1], temp[2], temp[3], temp[4], temp[5], temp[6], temp[7], temp[8]);
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

int getPossibleOutcomes(int8_t current_state[], int8_t* R_tensor[][27][27])
{
	int8_t R_matrix[9];
	memcpy(R_matrix, current_state, 9);
	for(int i = 0;i<9;i++)
	{
		if(current_state[i] != 0)
			continue;
		else
		{
			int8_t reward = 0;
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
			setRMatrix(current_state, R_tensor, R_matrix);
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

int8_t* getRMatrix(const int8_t current_state[], int8_t* R_tensor[][27][27])
{
	int hash = getHash(current_state);
	if((hash&0xff) == 0xff)
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
			int8_t byte = (*current_state)>>(8*(3*i+j));
			if(byte == 0xff)
				byte = 0;
			else if(byte == 0x01)
				byte = 1;
			else if (byte == 0x00)
				byte = 2;
			else
				return 0xffffffff;

			if(j==0)
				indices[i] += 3*3*byte;
			else if(j==1)
				indices[i] += 3*byte;
			else
				indices[i] += byte;
		}
	}

	return (int)(*indices);
}

bool setRMatrix(const int8_t current_state[], int8_t* R_tensor[][27][27], int8_t R_matrix[])
{
	int hash = getHash(current_state);
	if((hash&0xff) == 0xff)
		return false;
	R_tensor[hash>>24][hash>>16][hash>>8] = R_matrix;
	return true;
}
