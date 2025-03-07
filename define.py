# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    define.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 18:02:30 by aderouba          #+#    #+#              #
#    Updated: 2025/03/07 19:22:35 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# Input define
NUMBER_OF_COLUMN = 32

# Files defines
TRAINING_DATA_FILE = "trainingData.csv"
VALIDATION_DATA_FILE = "validationData.csv"
PARAMS_FILE = ".NNparams"

# Math define
EPSILON = 1e-15

# Neural network defines
INPUT_SIZE = NUMBER_OF_COLUMN - 2
OUTPUT_SIZE = 1
HIDDEN_LAYERS = [8, 8, 8]
SEED = None # None for random, number for set a seed

# Learning define
LEARNING_LOOP = 10000
LEARNING_RATE = 0.1
EARLY_SAME_LIMIT = 30
EARLY_SAME_PRECISION = 0.5

# Compare
COMPARE_LAYERS = [
    [4, 4],
    [8, 8],
    [16, 16],
    [32, 32],
    [4, 4, 4],
    [8, 8, 8],
    [16, 16, 16],
    [32, 32, 32],
    [4, 4, 4, 4],
    [8, 8, 8, 8],
    [16, 16, 16, 16],
    [32, 32, 32, 32],
]
