#pragma once

#include <stdbool.h>
#include <stdint.h>

// macro functions
#ifdef DEBUG 
	#define DEBUG_PRINT(fmt, ...) \
		do { if (debug_mode) printf(fmt, ##__VA_ARGS__); } while(0)
	#define DEBUG_EXEC(code) \
		do { if (debug_mode) { code; } } while(0)
#else
	#define DEBUG_PRINT(fmt, ...)
	#define DEBUG_EXEC(code)
#endif

#define P1 1
#define P2 -1
#define TIE 0
#define RFILE "R_tensor.dat"
#define QFILE "Q_tensor.dat"

// Error numbers
// EC: critical errors
#define EC_Q_MATRIX_ERROR 1
#define E_TIE_DETECTED 2
#define EC_ETC 3
#define EC_HASH_FAIL 4
#define EC_FILE_OPEN 5
#define EC_FILE_WRITE 6
#define EC_FILE_READ 7

void printHelp(const char *argv);
bool isGameover(const int8_t current_state[], int8_t player);
void getRValue(const int8_t current_state[], const int8_t R_tensor[][2], int8_t result[]);
void getQMatrix(const int8_t current_state[], const int Q_tensor[][9], int result[]);
int getHash(const int8_t current_state[]);
int unhash(const int *hash, int8_t state[]);
void setRValue(const int8_t current_state[], int8_t R_tensor[][2], int8_t R_value[]);
void setQMatrix(const int8_t current_state[], int Q_tensor[][9], int Q_matrix[]);
void printMatrix(const int8_t matrix[]);
void resetMatrix(void *matrix, int length);
void setErr(int num);
void resetErr();
bool verifyQMatrix(const int8_t current_state[], const int Q_matrix[]);
bool verifyGamma(const float gamma);
#ifdef DEBUG
void addQCount(const int8_t current_state[], int Q_update_count[]);
int getQCount(const int8_t current_state[], int Q_update_count[]);
void printQTensor(const int Q_tensor[][9]);
#endif
int chooseMove(const int8_t current_state[], const int Q_tensor[][9]);
int chooseOptimalSpot(const int8_t current_state[9], const int Q_matrix[9]);
int playGame(const int Q_tensor[][9]);
int chooseRandomEmpty(const int8_t current_state[]);
int saveRTensor(const int8_t R_tensor[][2], char* file_name);
int saveQTensor(const int Q_tensor[][9], char* file_name);
int saveTensors(const int Q_tensor[][9], char* Q_file, const int8_t R_tensor[][2], char* R_file);
int loadRTensor(int8_t R_tensor[][2], char* file_name);
int loadQTensor(int Q_tensor[][9], char* file_name);
int loadTensors(int Q_tensor[][9], char* Q_file, int8_t R_tensor[][2], char* R_file);
void detectError();
