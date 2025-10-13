#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common.h"
#include "training.h"

extern int errnum;
extern int debug_mode;

#ifdef DEBUG
int Q_update_count[27][27][27] = {0};
float avg_update_count = 0; // average number of updates chosen Q matrix has
int temporary_update_count_sum=0;
#endif

int trainMode(const int train_iteration, const float gamma, const int train_options,
			  const int game_options)
{
	srand(time(NULL));
	int8_t R_tensor[27][27][27][2];
	int Q_tensor[27][27][27][9];
	memset(R_tensor, 0xff, 27 * 27 * 27 * 2 * sizeof(int8_t));
	memset(Q_tensor, 0, 27 * 27 * 27 * 9 * sizeof(int));

	int8_t current_state[9] = {0};
	// create R tensor
#ifdef DEBUG
	if(generateRTensor(current_state, R_tensor) == -1)
	{
		printf("error generating R tensor\n");
		exit(-1);
	}
#else
	generateRTensor(current_state, R_tensor);
#endif

	int results[SEGMENTS][3];
	memset(results, 0, SEGMENTS * 3 * sizeof(int));
	// training session
	for(int i = 0; i < train_iteration; i++)
	{
#ifdef DEBUG
		int result = simulateGame(R_tensor, Q_tensor, gamma, train_options, game_options);
		if(errnum)
			return -1;
		else
			results[i * SEGMENTS / train_iteration][result + 1]++;
		if((i+1)%(train_iteration/SEGMENTS)==0)
		{
			DEBUG_PRINT("avg update count: %.3f\n", avg_update_count);
			avg_update_count= 0;
		}
#else
		simulateGame(R_tensor, Q_tensor, gamma, train_options, game_options);
#endif
	}
	//DEBUG_EXEC(printUpdateCount());
	//DEBUG_EXEC(printResults(results));

	return 0;
}

int trainQTensor(const int8_t current_state[], int Q_tensor[][27][27][9],
				 const int8_t R_tensor[][27][27][2], const float gamma, const int options)
{
	int next_move = chooseRandom(current_state);
	int8_t next_state[9];
	memcpy(next_state, current_state, 9 * sizeof(int8_t));
	next_state[next_move] = P1;

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

		updated_Q_value = (1 - gamma) * Q_current + next_move_reward + gamma * Q_max;
	}
	else if((options & TRAINQTENSOR_MODE) == TRAINQTENSOR_USEAVGQ)
	{
		int Q_avg = chooseAverageQValue(next_state, Q_tensor);
		if(errnum)
			return -1;

		updated_Q_value = (1 - gamma) * Q_current + next_move_reward + gamma * Q_avg;
	}

	Q_matrix[next_move] = updated_Q_value;
	setQMatrix(current_state, Q_tensor, Q_matrix);
#ifdef DEBUG
	if(debug_mode)
	{
		temporary_update_count_sum += getQCount(current_state, Q_update_count);
		addQCount(current_state, Q_update_count);
		if(errnum == EC_HASH_FAIL)
			return -1;
	}
	if(!verifyQMatrix(current_state, Q_matrix))
	{
		setErr(EC_Q_MATRIX_ERROR);
		DEBUG_EXEC(printMatrix(current_state));
		return -1;
	}
#endif

	return next_move;
}

int chooseMaxQValue(const int8_t current_state[], const int Q_tensor[][27][27][9])
{
	int Q_matrix[9];
	resetMatrix(Q_matrix, sizeof(int));
	getQMatrix(current_state, Q_tensor, Q_matrix);
	if(errnum)
		return -1;

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

int chooseAverageQValue(const int8_t current_state[], const int Q_tensor[][27][27][9])
{
	int Q_matrix[9];
	resetMatrix(Q_matrix, sizeof(int));
	getQMatrix(current_state, Q_tensor, Q_matrix);
	if(errnum)
		return -1;

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
			current_state[i] = P1;
			// set reward
			if(isGameover(current_state, P1))
				reward = 100;
			// set penalty
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
						current_state[j] = P2;
						if(isGameover(current_state, P2))
							reward -= 100;
						else
							generateRTensor(current_state, R_tensor);
						current_state[j] = 0;
					}
				}
				if(reward != 0)
					reward /= branch_count;
			}

			// save value
			current_state[i] = 0;
			R_value[0] = i;
			R_value[1] = reward;
			setRValue(current_state, R_tensor, R_value);
			if(errnum)
				return -1;

			// reset matrices
			memset(R_value, 0, 2);
		}
	}

	return 0;
}

int chooseMove(const int8_t current_state[], const int8_t R_tensor[][27][27][2])
{
	int8_t R_value[2];
	getRValue(current_state, R_tensor, R_value);

	if(R_value[1] <= 0)
	{
	}

	return 0;
}

int simulateGame(const int8_t R_tensor[][27][27][2], int Q_tensor[][27][27][9], const float gamma,
				 const int train_options, const int game_options)
{
	int8_t current_state[9] = {0};
	int next_move;
	int return_value;
#ifdef DEBUG
	int move_count=0;
#endif
	if((game_options & SIMULATEGAME_OPTIONS) == SIMULATEGAME_RANDOMSTART)
	{
		int r = rand() % 9;
		current_state[r] = 1;
#ifdef DEBUG
		move_count++;
#endif
	}
	else
	{
		// player move
		next_move = trainQTensor(current_state, Q_tensor, R_tensor, gamma, train_options);
		if(errnum)
			return -1;
		current_state[next_move] = P1;
#ifdef DEBUG
		move_count++;
#endif
	}
	while(1)
	{
		// check for tie
		next_move = chooseRandom(current_state);
		if(errnum == E_TIE_DETECTED)
		{
			resetErr();
			return_value = 0;
			break;
		}
		else if(errnum == EC_ETC)
			return -1;

		// opponent next move
		current_state[next_move] = P2;
#ifdef DEBUG
		move_count++;
#endif
		if(isGameover(current_state, P2))
			return_value = P2;

		// player move
		next_move = trainQTensor(current_state, Q_tensor, R_tensor, gamma, train_options);
		if(errnum)
			return -1;
		current_state[next_move] = P1;
#ifdef DEBUG
		move_count++;
#endif
		if(isGameover(current_state, P1))
			return_value = P1;
	}

#ifdef DEBUG
	avg_update_count += (float)temporary_update_count_sum/move_count;
	temporary_update_count_sum = 0;
#endif

	return return_value;
}

void printResults(const int results[][3])
{
	for(int i = 0; i < SEGMENTS; i++)
	{
		int game_count = results[i][0] + results[i][1] + results[i][2];
		printf("Segment [%d]:\n", i);
		printf("%10s : %3d\t%9s : %.3f\n", "win count", results[i][2], "win count",
			   (float)results[i][2] / game_count);
		printf("%10s : %3d\t%9s : %.3f\n", "tie count", results[i][1], "tie count",
			   (float)results[i][1] / game_count);
		printf("%10s : %3d\t%9s : %.3f\n", "lose count", results[i][0], "lose count",
			   (float)results[i][0] / game_count);
	}
}

#ifdef DEBUG
void printRTensor(const int8_t R_tensor[][27][27][2])
{
	for(int i = 0;i<27;i++)
	{
		for(int j = 0;j<27;j++)
		{
			for(int k = 0;k<27;k++)
			{
				if((int8_t)R_tensor[i][j][k][0] != -1)
					printf("%d: %d\n", R_tensor[i][j][k][0], R_tensor[i][j][k][1]);
			}
		}
	}
}

void printUpdateCount()
{
	for(int i = 0;i<27;i++)
	{
		for(int j = 0;j<27;j++)
		{
			for(int k = 0;k<27;k++)
			{
				if(Q_update_count[i][j][k] != 0)
					printf("update counts: %d\n", Q_update_count[i][j][k]);
			}
		}
	}
}
#endif
