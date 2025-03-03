# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    define.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 18:02:30 by aderouba          #+#    #+#              #
#    Updated: 2025/03/03 15:43:55 by gugus            ###   ########.fr        #
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
HIDDEN_LAYERS = [32, 32, 32]

# Learning define
LEARNING_LOOP = 2000
LEARNING_RATE = 0.1
