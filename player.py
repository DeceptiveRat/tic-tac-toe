#!/bin/python3 

import sys
import getopt
import numpy as np
import pdb

# global variables
end_conditions = []
x_len = 3
y_len = 3
iterations=0

def usage():
	print("usage:", sys.argv[0])
	print("options:")
	print("-h: display this help screen")
	print("-t: train model")

def get_possible_outcomes(input_matrix):
	global iterations
	iterations += 1
	print(iterations)
	return_value = []
	input_matrix = input_matrix.reshape(-1)
	empty_spots = np.where(input_matrix == 0)
	for player_choice in empty_spots[0]:
		reward = 0
		# player turn
		input_matrix[player_choice] = 1
		if is_gameover(input_matrix.reshape(x_len, y_len)):
			reward = 100

		else:
			# opponent turn
			for opponent_choice in empty_spots[0]:
				if opponent_choice == player_choice:
					continue

				input_matrix[opponent_choice] = -1
				if is_gameover(input_matrix.reshape(x_len, y_len)):
					reward -= 100/(len(empty_spots[0]) -1)
				else:
					return_value += get_possible_outcomes(input_matrix.reshape(x_len, y_len))
				input_matrix[opponent_choice] = 0
		
		input_matrix[player_choice] = reward
		return_value.append(np.copy(input_matrix.reshape(x_len, y_len)))
		input_matrix[player_choice] = 0
	
	return return_value

def is_gameover(input_matrix):
	# win
	boolean_matrix = (input_matrix == 1)
	for _ in end_conditions:
		if np.array_equal(boolean_matrix & _, _):
			return True
	
	# lose
	boolean_matrix = (input_matrix == -1)
	for _ in end_conditions:
		if np.array_equal(boolean_matrix & _, _):
			return True
	
	return False

try:
	opts, args = getopt.getopt(sys.argv[1:], "ht")
except getopt.GetoptError as err:
	print(err)
	usage()
	sys.exit(2)

mode = ""

for option, argument in opts:
	if option == "-h":
		usage()
		sys.exit()
	elif option == "-t":
		mode = "train"
	else:
		assert False, "unhandled option"

if mode == "":
	print("choose mode!")
	usage()
	sys.exit(2)

if mode == "train":
	# === create R tensor using DP ===
	board = np.zeros((3, 3), dtype=int)

	end_conditions.append(np.array([[True, False, False], [False, True, False], [False, False, True]]))
	end_conditions.append(np.array([[True, True, True], [False, False, False], [False, False, False]]))
	end_conditions.append(np.array([[False, False, False], [True, True, True], [False, False, False]]))
	end_conditions.append(np.array([[False, False, False], [False, False, False], [True, True, True]]))
	end_conditions.append(np.array([[True, False, False], [True, False, False], [True, False, False]]))
	end_conditions.append(np.array([[False, True, False], [False, True, False], [False, True, False]]))
	end_conditions.append(np.array([[False, False, True], [False, False, True], [False, False, True]]))
	end_conditions.append(np.array([[False, False, True], [False, True, False], [True, False, False]]))

	outcomes = get_possible_outcomes(board)
	print(len(outcomes))
