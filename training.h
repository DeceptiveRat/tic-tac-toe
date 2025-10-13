#pragma once

#include <stdlib.h>

#define SEGMENTS 4

// ===== options =====
// simulateGame options
#define SIMULATEGAME_OPTIONS 0xff
#define SIMULATEGAME_RANDOMSTART 0x01

// trainQTensor options
#define TRAINQTENSOR_MODE 0xff
#define TRAINQTENSOR_USEMAXQ 0x01
#define TRAINQTENSOR_USEAVGQ 0x02

// === end options ===

int trainMode(const int train_iteration, const float gamma, const int train_options,
			  const int game_options);
int trainQTensor(const int8_t current_state[], int Q_tensor[][27][27][9],
				 const int8_t R_tensor[][27][27][2], const float gamma, const int options);
int generateRTensor(int8_t current_state[], int8_t R_tensor[][27][27][2]);
int chooseRandom(const int8_t current_state[]);
int chooseMaxQValue(const int8_t current_state[], const int Q_tensor[][27][27][9]);
int chooseAverageQValue(const int8_t current_state[], const int Q_tensor[][27][27][9]);
int simulateGame(const int8_t R_tensor[][27][27][2], int Q_tensor[][27][27][9], const float gamma,
				 const int train_options, const int game_options);
void printResults(const int results[][3]);
#ifdef DEBUG
void printRTensor(const int8_t R_tensor[][27][27][2]);
void printUpdateCount();
#endif
