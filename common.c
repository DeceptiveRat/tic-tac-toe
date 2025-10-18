#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

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
	printf("-s: save trained tensors (file names: %s, %s)\n", QFILE, RFILE);
	printf("-l: load trained tensors (file names: %s, %s)\n", QFILE, RFILE);
	printf("-p: train player 2 (can be used with -o)\n");
	printf("-o: train player 2 (can be used with -p)\n");
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

void getRValue(const int8_t current_state[], const int8_t R_tensor[][2], int8_t result[])
{
	int hash = getHash(current_state);
	if(hash >= 19683)
	{
		setErr(EC_HASH_FAIL);
		return;
	}

	memcpy(result, R_tensor[hash], 2);
	return;
}

void getQMatrix(const int8_t current_state[], const int Q_tensor[][9], int result[])
{
	int hash = getHash(current_state);
	if(hash >= 19683)
	{
		setErr(EC_HASH_FAIL);
		return;
	}

	memcpy(result, Q_tensor[hash], 9 * sizeof(int));
	return;
}

int getHash(const int8_t current_state[])
{
	int hash = 0;
	for(int i= 0;i<9;i++)
		hash += (current_state[i] + 1)*(int)pow(3, i);
	
	return hash;

}

int unhash(const int *hash, int8_t state[])
{
	for(int i = 0;i<9;i++)
		state[i] = (*hash/(int)pow(3, i))%3;

	return 0;
}

void setRValue(const int8_t current_state[], int8_t R_tensor[][2], int8_t R_value[])
{
	int hash = getHash(current_state);
	if(hash >= 19683)
	{
		setErr(EC_HASH_FAIL);
		return;
	}
	memcpy(R_tensor[hash], R_value, 2);
	return;
}

void setQMatrix(const int8_t current_state[], int Q_tensor[][9], int Q_matrix[])
{
	int hash = getHash(current_state);
	if(hash >= 19683)
	{
		setErr(EC_HASH_FAIL);
		return;
	}
	memcpy(Q_tensor[hash], Q_matrix, 9 * sizeof(int));
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

#ifdef DEBUG
void addQCount(const int8_t current_state[], int Q_update_count[])
{
	int hash = getHash(current_state);
	if(hash >= 19683)
	{
		setErr(EC_HASH_FAIL);
		return;
	}
	Q_update_count[hash]++;
	return;
}

int getQCount(const int8_t current_state[], int Q_update_count[])
{
	int hash = getHash(current_state);
	if(hash >= 19683)
	{
		setErr(EC_HASH_FAIL);
		return -1;
	}
	return Q_update_count[hash];
}
#endif

int chooseMove(const int8_t current_state[], const int Q_tensor[][9])
{
	int Q_matrix[9];
	getQMatrix(current_state, Q_tensor, Q_matrix);
	if(errnum == EC_HASH_FAIL)
		return -1;

	int max_index = 0;
	for(int i = 1;i<9;i++)
	{
		if(Q_matrix[i] > Q_matrix[max_index])
			max_index = i;
	}

	return max_index;
}

int playGame(const int Q_tensor[][9])
{
	int8_t current_state[9] = {0};
	int Q_matrix[9];
	int choice;

	while(1)
	{
		choice = 0;
		// P1 move
		getQMatrix(current_state, Q_tensor, Q_matrix);
		if(errnum == EC_HASH_FAIL)
			return -1;
		choice = chooseOptimalSpot(current_state, Q_matrix);
		if(errnum == E_TIE_DETECTED)
		{
			resetErr();
			return TIE;
		}
		current_state[choice] = P1;
		if(isGameover(current_state, P1) == true)
			return P1;

		// P2 move
		getQMatrix(current_state, Q_tensor, Q_matrix);
		if(errnum == EC_HASH_FAIL)
			return -1;
		choice = chooseOptimalSpot(current_state, Q_matrix);
		current_state[choice] = P2;
		if(isGameover(current_state, P2) == true)
			return P2;
	}
}

int chooseRandomEmpty(const int8_t current_state[])
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
		setErr(E_TIE_DETECTED);
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
	setErr(EC_ETC);
	return -1;
}

int chooseOptimalSpot(const int8_t current_state[9], const int Q_matrix[9])
{
	int optimal_index=-1;
	for(int i = 0;i<9;i++)
	{
		if(current_state[i] == 0)
		{
			if(optimal_index == -1)
			{
				optimal_index = i;
				continue;
			}

			if(Q_matrix[optimal_index] < Q_matrix[i])
				optimal_index = i;
			// choose randomly if both have same value
			else if(Q_matrix[optimal_index] == Q_matrix[i])
				optimal_index = (rand()%2==0)?(optimal_index):(i);
		}
	}

	if(optimal_index == -1)
		setErr(E_TIE_DETECTED);
	return optimal_index;
}

void printQTensor(const int Q_tensor[][9])
{
	for(int i = 0;i<19683;i++)
	{
		int empty[9] = {0};
		int temp[9];
		memcpy(temp, Q_tensor[i], 9*sizeof(int));
		if(memcmp(empty, temp, 9*sizeof(int)) != 0)
		{
			printf("===============================\n");
			printf("%-3d %8d %8d\n%-3d %8d %8d\n%-3d %8d %8d\n", temp[0], temp[1], temp[2], temp[3], temp[4], temp[5], temp[6], temp[7], temp[8]);
		}
	}
}

int saveRTensor(const int8_t R_tensor[][2], char* file_name)
{
	FILE* file_ptr;
	file_ptr = fopen(file_name, "wb");
	if(file_ptr == NULL)
	{
		setErr(EC_FILE_OPEN);
		return -1;
	}

	if(fwrite(R_tensor, sizeof(int), 19683*2, file_ptr) != 19683*2)
	{
		setErr(EC_FILE_WRITE);
		return -1;
	}

	fclose(file_ptr);
	return 0;
}

int saveQTensor(const int Q_tensor[][9], char* file_name)
{
	FILE* file_ptr;
	file_ptr = fopen(file_name, "wb");
	if(file_ptr == NULL)
	{
		setErr(EC_FILE_OPEN);
		return -1;
	}

	if(fwrite(Q_tensor, sizeof(int), 19683*9, file_ptr) != 19683*9)
	{
		setErr(EC_FILE_WRITE);
		return -1;
	}

	fclose(file_ptr);
	return 0;
}

int saveTensors(const int Q_tensor[][9], char* Q_file, const int8_t R_tensor[][2], char* R_file)
{
	saveRTensor(R_tensor, R_file);
	if(errnum)
		return -1;
	saveQTensor(Q_tensor, Q_file);
	if(errnum)
		return -1;
	return 0;
}

int loadRTensor(int8_t R_tensor[][2], char* file_name)
{
	FILE* file_ptr;
	file_ptr = fopen(file_name, "rb");
	if(file_ptr == NULL)
	{
		setErr(EC_FILE_OPEN);
		return -1;
	}

	if(fread(R_tensor, sizeof(int), 19683*2, file_ptr) != 19683*2)
	{
		setErr(EC_FILE_READ);
		return -1;
	}

	fclose(file_ptr);
	return 0;
}

int loadQTensor(int Q_tensor[][9], char* file_name)
{
	FILE* file_ptr;
	file_ptr = fopen(file_name, "rb");
	if(file_ptr == NULL)
	{
		setErr(EC_FILE_OPEN);
		return -1;
	}

	if(fread(Q_tensor, sizeof(int), 19683*9, file_ptr) != 19683*9)
	{
		setErr(EC_FILE_READ);
		return -1;
	}

	fclose(file_ptr);
	return 0;
}

int loadTensors(int Q_tensor[][9], char* Q_file, int8_t R_tensor[][2], char* R_file)
{
	loadRTensor(R_tensor, R_file);
	if(errnum)
		return -1;
	loadQTensor(Q_tensor, Q_file);
	if(errnum)
		return -1;
	return 0;
}

void detectError()
{
	if(errnum)
	{
		printf("Critical Error!\n");
		printf("Error number: %d\n", errnum);
		exit(-1);
	}
}
