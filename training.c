#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>

#include "common.h"
#include "training.h"

extern int errnum;
extern int debug_mode;

int trainMode(int Q_tensor[][9], int8_t R_tensor[][2], const int train_iteration,
			  const float gamma, const int train_options, const int game_options)
{

	int8_t current_state[9] = {0};
	// create R tensor
#ifdef DEBUG
	if(generateRTensor(current_state, R_tensor, P1) == -1)
	{
		printf("error generating R tensor\n");
		exit(-1);
	}
#else
	generateRTensor(current_state, R_tensor, P1);
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
#else
		simulateGame(R_tensor, Q_tensor, gamma, train_options, game_options);
#endif
	}

	return 0;
}

int trainQTensor(const int8_t current_state[], int Q_tensor[][9],
				 const int8_t R_tensor[][2], const float gamma, const int options, const int turn)
{
	// choose next move randomly
	int next_move = chooseRandomEmpty(current_state);
	if(errnum == E_TIE_DETECTED)
		return -1;
	int8_t next_state[9];
	memcpy(next_state, current_state, 9 * sizeof(int8_t));
	next_state[next_move] = turn;

	// get reward for next move
	int8_t R_value[2];
	getRValue(current_state, R_tensor, R_value);
	if(errnum == EC_HASH_FAIL)
		return -1;
	int next_move_reward = (R_value[0] == next_move) ? (R_value[1]) : (0);

	// get Q matrix
	int Q_matrix[9];
	getQMatrix(current_state, Q_tensor, Q_matrix);
	if(errnum == EC_HASH_FAIL)
		return -1;
	int Q_current = Q_matrix[next_move];

	// update Q value
	int updated_Q_value;
	if((options & TRAINQTENSOR_MODE) == TRAINQTENSOR_USEMAXQ)
	{
		// choose potential max reward if we make the move
		int Q_max = chooseMaxQValue(next_state, Q_tensor, turn);
		if(errnum)
		{
			if(errnum == E_TIE_DETECTED)
			{
				Q_max = 0;
				resetErr();
			}
			else
				return -1;
		}

		updated_Q_value = (1 - gamma) * Q_current + next_move_reward + gamma * Q_max;
	}
	else if((options & TRAINQTENSOR_MODE) == TRAINQTENSOR_USEAVGQ)
	{
		// choose average of potential reward if we make the move
		int Q_avg = chooseAverageQValue(next_state, Q_tensor, turn);
		if(errnum)
		{
			if(errnum == E_TIE_DETECTED)
			{
				Q_avg = 0;
				resetErr();
			}
			else
				return -1;
		}

		updated_Q_value = (1 - gamma) * Q_current + next_move_reward + gamma * Q_avg;
	}
	
	// save updated Q value
	Q_matrix[next_move] = updated_Q_value;
	setQMatrix(current_state, Q_tensor, Q_matrix);
#ifdef DEBUG
	if(!verifyQMatrix(current_state, Q_matrix))
	{
		setErr(EC_Q_MATRIX_ERROR);
		DEBUG_EXEC(printMatrix(current_state));
		return -1;
	}
#endif

	return next_move;
}

int getMaxQValue(const int8_t current_state[], const int Q_tensor[][9])
{
	int Q_matrix[9];
	resetMatrix(Q_matrix, sizeof(int));
	getQMatrix(current_state, Q_tensor, Q_matrix);
	if(errnum)
		return -1;

	int max = INT_MIN;
	for(int i = 0; i < 9; i++)
	{
		// not available spot
		if(current_state[i] != 0)
			continue;
		if(Q_matrix[i] > max)
			max = Q_matrix[i];
	}

	if(max == INT_MIN)
		DEBUG_PRINT("Not a tie but minimum value detected!\n");

	return max;
}

int chooseMaxQValue(const int8_t current_state[], const int Q_tensor[][9], int turn)
{
	// check for tie
	int empty_count = 0;
	for(int i = 0; i < 9; i++)
	{
		if(current_state[i] == 0)
			empty_count++;
	}
	if(empty_count == 0)
	{
		setErr(E_TIE_DETECTED);
		return -1;
	}

	int8_t new_state[9];
	memcpy(new_state, current_state, 9*sizeof(int8_t));
	int opponent = 0 - turn;
	int max = INT_MIN;

	// find max for each move of opponent
	for(int i =0;i<9;i++)
	{
		if(new_state[i] != 0)
			continue;

		new_state[i] = opponent;
		int result = getMaxQValue(new_state, Q_tensor);
		if(result > max)
			max = result;
		new_state[i] = 0;
	}

	return max;
}

int chooseAverageQValue(const int8_t current_state[], const int Q_tensor[][9], int turn)
{
	// check for tie
	int empty_count = 0;
	for(int i = 0; i < 9; i++)
	{
		if(current_state[i] == 0)
			empty_count++;
	}
	if(empty_count == 0)
	{
		setErr(E_TIE_DETECTED);
		return -1;
	}

	int8_t new_state[9];
	memcpy(new_state, current_state, 9*sizeof(int8_t));
	int opponent = 0 - turn;
	int sum = 0;

	// find max for each move of opponent
	for(int i =0;i<9;i++)
	{
		if(new_state[i] != 0)
			continue;

		new_state[i] = opponent;
		sum += getMaxQValue(new_state, Q_tensor);
		new_state[i] = 0;
	}

	return sum/empty_count;
}

int generateRTensor(int8_t current_state[], int8_t R_tensor[][2], const int turn)
{
	int8_t R_value[2];
	memset(R_value, 0, 2);
	for(int i = 0; i < 9; i++)
	{
		if(current_state[i] != 0)
			continue;

		int reward = 0;
		current_state[i] = turn;
		// set reward
		if(isGameover(current_state, turn))
			reward = 100;
		else
		{
			reward = 0;
			generateRTensor(current_state, R_tensor, 0-turn);
		}

		// revert state
		current_state[i] = 0;
		// save value
		R_value[0] = i;
		R_value[1] = reward;
		setRValue(current_state, R_tensor, R_value);
		if(errnum)
			return -1;

		// reset matrices
		memset(R_value, 0, 2);
	}

	return 0;
}

int simulateGame(const int8_t R_tensor[][2], int Q_tensor[][9], const float gamma,
				 const int train_options, const int game_options)
{
	int8_t current_state[9] = {0};
	int next_move;
	int return_value;
	if(game_options & SIMULATEGAME_RANDOMSTART)
	{
		int r = rand() % 9;
		current_state[r] = 1;
	}
	else
	{
		// P1 move
		if(game_options & SIMULATEGAME_PLAYERP1)
			next_move = trainQTensor(current_state, Q_tensor, R_tensor, gamma, train_options, P1);
		else
			next_move = chooseRandomEmpty(current_state);
		if(errnum)
			return -1;
		current_state[next_move] = P1;
	}
	while(1)
	{
		// P2 next move
		if(game_options & SIMULATEGAME_PLAYERP2)
			next_move = trainQTensor(current_state, Q_tensor, R_tensor, gamma, train_options, P2);
		else
			next_move = chooseRandomEmpty(current_state);
		if(errnum == E_TIE_DETECTED)
		{
			resetErr();
			return_value = 0;
			break;
		}
		current_state[next_move] = P2;
		if(isGameover(current_state, P2))
			return_value = P2;

		// P1 move
		if(game_options & SIMULATEGAME_PLAYERP1)
			next_move = trainQTensor(current_state, Q_tensor, R_tensor, gamma, train_options, P1);
		else
			next_move = chooseRandomEmpty(current_state);
		if(errnum == E_TIE_DETECTED)
		{
			resetErr();
			return_value = 0;
			break;
		}
		current_state[next_move] = P1;
		if(isGameover(current_state, P1))
			return_value = P1;
	}

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
void printRTensor(const int8_t R_tensor[][2])
{
	for(int i = 0; i < 19683; i++)
		if((int8_t)R_tensor[i][0] != -1)
			printf("%d: %d\n", R_tensor[i][0], R_tensor[i][1]);
}
#endif
