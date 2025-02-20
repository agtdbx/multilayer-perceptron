# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    file.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aderouba <aderouba@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/13 17:27:19 by aderouba          #+#    #+#              #
#    Updated: 2025/02/20 17:46:51 by aderouba         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from parsing.print import printError

def readFile(path: str) -> str:
    """
    Function for read all content in a file.
    Print it's error.
    @param  path    The file path to read.
    @return The file content, or None in case of error.
    """
    file = None

    # Open file
    try:
        file = open(path, "r")
    except:
        printError(f"can't open file '{path}'")
        return None

    # Read file
    content = ""
    try:
        content = file.read()
    except:
        printError(f"can't read file '{path}'")
        content = None

    # close file
    file.close()

    return content


def writeInFile(path: str, content: str) -> bool:
    """
    Function for write content in a file.
    Print it's error.
    @param  path    The file path to write.
    @param  content The content to write in the file.
    @return True in case of success, False else.
    """
    file = None

    # Open file
    try:
        file = open(path, "w")
    except:
        printError(f"can't open file '{path}'")
        return False

    # Read file
    try:
        file.write(content)
    except:
        printError(f"can't write in file '{path}'")
        return False

    # close file
    file.close()

    return True
