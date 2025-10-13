#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "common.h"

int errnum = 0;
int debug_mode=0;

void setErr(int num)
{
	if(num == 0)
		return;
	errnum = num;
}

void resetErr() { errnum = 0; };

bool verifyQMatrix(const int8_t current_state[], const int Q_matrix[])
{
	for(int i = 0; i < 9; i++)
	{
		if(Q_matrix[i] != 0)
		{
			// Q matrix filling non-empty spot
			if(current_state[i] != 0)
				return false;
		}
	}

	return true;
}

void printHelp(const char *argv)
{
	printf("Usage: %s\n", argv);
	printf("options:");
	printf("-h: for this help message\n");
	printf("-t: to train model (default)\n");
	printf("-d: set to debug mode. Must compile with -DDEBUG!\n");
	printf("-c <int>: to set train iteration count (100 by default)\n");
	printf("-g <float>: set gamma (0.5 by default)\n");
	printf("-m: [train mode] train Q matrix with max values (default)(do not use with other "
		   "train modes!)\n");
	printf("-a: [train mode] train Q matrix with avg values (do not use with other train "
		   "modes!)\n");
	printf("-r: [game options] Player starts at random location (default)(do not use with other "
		   "game options!)\n");
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

void getRValue(const int8_t current_state[], const int8_t R_tensor[][27][27][2], int8_t result[])
{
	int hash = getHash(current_state);
	if((hash & 0xff000000) != 0x00)
	{
		setErr(E_CRITICAL_ERROR);
		return;
	}

	memcpy(result, R_tensor[hash >> 24][hash >> 16][hash >> 8], 2);
	return;
}

void getQMatrix(const int8_t current_state[], const int Q_tensor[][27][27][9], int result[])
{
	int hash = getHash(current_state);
	if((hash & 0xff000000) != 0x00)
	{
		setErr(E_CRITICAL_ERROR);
		return;
	}

	memcpy(result, Q_tensor[hash >> 24][hash >> 16][hash >> 8], 9 * sizeof(int));
	return;
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

void setRValue(const int8_t current_state[], int8_t R_tensor[][27][27][2], int8_t R_value[])
{
	int hash = getHash(current_state);
	if((hash & 0xff000000) != 0x00)
	{
		setErr(E_CRITICAL_ERROR);
		return;
	}
	memcpy(R_tensor[(hash >> 16) & 0xff][(hash >> 8) & 0xff][(hash) & 0xff], R_value, 2);
	return;
}

void setQMatrix(const int8_t current_state[], int Q_tensor[][27][27][9], int Q_matrix[])
{
	int hash = getHash(current_state);
	if((hash & 0xff000000) != 0x00)
	{
		setErr(E_CRITICAL_ERROR);
		return;
	}
	memcpy(Q_tensor[(hash >> 16) & 0xff][(hash >> 8) & 0xff][(hash) & 0xff], Q_matrix,
		   9 * sizeof(int));
	return;
}

void printMatrix(const int8_t matrix[])
{
	printf("printing matrix...\n");
	printf("1: %-3d %8d %8d\n2: %-3d %8d %8d\n3: %-3d %8d %8d\n", matrix[0], matrix[1], matrix[2],
		   matrix[3], matrix[4], matrix[5], matrix[6], matrix[7], matrix[8]);
}

void resetMatrix(void *matrix, int length) { memset(matrix, 0, 9 * length); }

bool verifyGamma(const float gamma) { return ((gamma >= 0) && (gamma <= 1)) ? (true) : (false); }
