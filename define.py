# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    define.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 18:02:30 by aderouba          #+#    #+#              #
#    Updated: 2025/02/25 14:34:03 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# Input define
NUMBER_OF_COLUMN = 32
INPUT_SIZE = NUMBER_OF_COLUMN - 2

# Files defines
TRAINING_DATA_FILE = "trainingData.csv"
VALIDATION_DATA_FILE = "validationData.csv"
PARAMS_FILE = ".NNparams"

# Neural network defines
LAYERS = [INPUT_SIZE, 8, 8, 2]
LEARNING_LOOP = 10
LEARNING_RATE = 1
