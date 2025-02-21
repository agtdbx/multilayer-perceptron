# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    define.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 18:02:30 by aderouba          #+#    #+#              #
#    Updated: 2025/02/21 17:27:06 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# Input define
NUMBER_OF_COLUMN = 32

# Files defines
TRAINING_DATA_FILE = "trainingData.csv"
VALIDATION_DATA_FILE = "trainingData.csv"
PARAMS_FILE = ".NNparams"

# Neural network defines
LAYERS = [NUMBER_OF_COLUMN - 1, 8, 8, 1]
ERROR_PRECISION = 0.1
