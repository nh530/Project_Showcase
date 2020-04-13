"""
--------------
Specifications
--------------
-Task (T): Playing  Tic-Tac-Toe
-Perfromance(P): % of games won in Tournament
-Experience: Data from games played against self (Indirect feedback)
-Learning: It boils down to optimizing the function (Target function) that finds best Move
for a board State from a set of legal Moves.
-Experiment generator: creates the empty starting board of Tic-Tac-Toe. It generates the training scenario of the task.
-Performance System: Takes as input the problem created by experiment generator and produces a step-by-step history
of the game. Two cloned AI based on the same estimate of the true model(target function) make decisions.
-Critic: Takes the solution trace of a game and outputs a set of training instances. Extracts features from game history
-Generalizer: Uses training examples provided by critic to update the target function. In this case, update weights.
---------------------------------
Design choices for Implementation
---------------------------------
Target function, V, is defined to be some function that describes changes to the target variable.
Target function is the function that maps board to the target variable. Target variable is assumed to be some real
number that describes the outcome of the event.  Target variable is 100 if player wins, -100 if player loses, 0 if draw,
and Y = V(X) = W*X if neither of the previous 2 conditions are satisfied. In other words, the relationship between
board features and the target is assumed to be linear.
-Approximate Target Function (V): Maps board -> R
-Let W denote the vector of weights
-Let X denote the vector of features that are extracted from a given board b_t, where b_t denotes the board
at turn t.
-Since we assume the underlying model is linear, we use mean squared error loss function.
-At each training iteration, the target variables are generated from previous learned weights.
-Each component in X captures a different property of the Tic-Tac-Toe board. For instance, $X_1$ is the number of
player1 letters in a row with 2 empty spots, $X_3$ is the number of player1 letters in a row with 1 empty spot, and
$X_5$ is the number of player1 letters in a row with no empty spots. Likewise, $X_2, X_4, X_6$ is the same but for
player2 letters.
-The algorithm should learn to place more weight to the features that indicate we're closer to a win.

TODO: Training the agent on 10,000 games create a sufficient AI that will always draw the player. However, if the player
TODO: gives the agent a free win, the agent will not take the free win.
"""
import numpy as np
import random
import copy


class ExperimentGenerator(object):
    """
    Experiment Generator generates New problems. In this case it just returns the same initial game boards.
    """

    def __init__(self):
        self.init_board_state = [' '] * 10  # Index 0 is not used.

    def generate_new_problem(self):
        """
        Returns initial Board State
        """
        return self.init_board_state


class Player(object):
    """
    Provides all the methods that a Player must execute in order to play a game of Tic-Tac-Toe
    """

    def __init__(self, player_target_function_weight_vector, player_symbol=None):
        self.player_symbol = player_symbol  # "X" or "O"
        self.player_target_function_weight_vector = player_target_function_weight_vector

    def is_game_over(self, board, player_symbol):
        """ Returns True if game is over else returns false """

        flag = False
        # Game already over
        if board == -1:
            flag = True
        # Game won by either player
        elif ((board[7] == board[8] == board[9] == player_symbol) or  # Across the bottom.
              (board[4] == board[6] == board[5] == player_symbol) or  # Across the middle.
              (board[1] == board[2] == board[3] == player_symbol) or  # Across the top.
              (board[7] == board[4] == board[1] == player_symbol) or  # Across the left side.
              (board[8] == board[5] == board[2] == player_symbol) or  # Down the middle.
              (board[9] == board[6] == board[3] == player_symbol) or  # Down the right side.
              (board[7] == board[5] == board[3] == player_symbol) or  # Diagonal
              (board[9] == board[5] == board[1] == player_symbol)):  # Diagonal
            flag = True
            print("Game is over!")
        # Board Full
        elif ' ' not in board[1:]:  # 0th index always left empty.
            flag = True
        return flag

    def look_for_legal_moves(self, board_state, player_symbol):
        """
        Returns a list of legal moves for a given board state.
        :params: board_state: representation of current board.
        :params: player_symbol: symbol denotes the player. 'X' or 'O'.
        :return: legal_moves: a list of boards with the next possible move, $b_{t+1}$.
        """

        legal_moves = []  # Keeps track of the board with new legal move.
        legal_moves = []  # keeps track of the index. For use in tic_tac_toe_emulator.py
        for i in range(1, 10):  # Iterating from 1 to 9
            if board_state[i] == ' ':  # If the position is empty.
                legal_moves.append(i)
                temp_board = copy.deepcopy(board_state)  # Create a copy of the board.
                temp_board[i] = player_symbol  # make the move.
                legal_moves.append(temp_board)  # append the board with the new move.
        return legal_moves, legal_moves

    def extractFeatures(self, board, playerSymbol1, playerSymbol2):
        """
        Returns extracted feature Vector for a given board state where the features take on values from 0 to 3
        :params: board: layout of the board.
        :params: playerSymbol1: "X" or "O" representing the player. (self)
        :params: playerSymbol2: "X" or "O" representing the other player.
        :returns: feature_vector:
        """
        w1, w2, w3, w4, w5, w6, w7, w8 = 0, 0, 0, 0, 0, 0, 0, 0  # Initializing feature vector.
        # if 1 of Player-1's Symbol in a row with two open spots, above or below.
        if (((board[1] == playerSymbol1) and (board[4] == board[7] == ' ')) or
                ((board[7] == playerSymbol1) and (board[4] == board[1] == ' ')) or
                ((board[4] == playerSymbol1) and (board[1] == board[7] == ' '))):
            w1 = w1 + 1
        if (((board[2] == playerSymbol1) and (board[5] == board[8] == ' ')) or
                ((board[8] == playerSymbol1) and (board[2] == board[5] == ' ')) or
                ((board[5] == playerSymbol1) and (board[2] == board[8] == ' '))):
            w1 = w1 + 1
        if (((board[3] == playerSymbol1) and (board[6] == board[9] == ' ')) or
                ((board[9] == playerSymbol1) and (board[6] == board[3] == ' ')) or
                ((board[6] == playerSymbol1) and (board[3] == board[9] == ' '))):
            w1 = w1 + 1

        # if 1 of Player-2's Symbol in a row with 2 open spots, above or below.
        if (((board[1] == playerSymbol2) and (board[4] == board[7] == ' ')) or
                ((board[7] == playerSymbol2) and (board[4] == board[1] == ' ')) or
                ((board[4] == playerSymbol2) and (board[1] == board[7] == ' '))):
            w2 = w2 + 1
        if (((board[2] == playerSymbol2) and (board[5] == board[8] == ' ')) or
                ((board[8] == playerSymbol2) and (board[2] == board[5] == ' ')) or
                ((board[5] == playerSymbol2) and (board[2] == board[8] == ' '))):
            w2 = w2 + 1
        if (((board[3] == playerSymbol2) and (board[6] == board[9] == ' ')) or
                ((board[9] == playerSymbol2) and (board[6] == board[3] == ' ')) or
                ((board[6] == playerSymbol2) and (board[3] == board[9] == ' '))):
            w2 = w2 + 1

        # if 2 of Player-2's Symbol in a row with 1 open spots, above or below.
        if (((board[1] == ' ') and (board[4] == board[7] == playerSymbol2)) or
                ((board[7] == ' ') and (board[4] == board[1] == playerSymbol2)) or
                ((board[4] == ' ') and (board[1] == board[7] == playerSymbol2))):
            w4 = w4 + 1
        if (((board[2] == ' ') and (board[5] == board[8] == playerSymbol2)) or
                ((board[8] == ' ') and (board[2] == board[5] == playerSymbol2)) or
                ((board[5] == ' ') and (board[2] == board[8] == playerSymbol2))):
            w4 = w4 + 1
        if (((board[3] == ' ') and (board[6] == board[9] == playerSymbol2)) or
                ((board[9] == ' ') and (board[6] == board[3] == playerSymbol2)) or
                ((board[6] == ' ') and (board[3] == board[9] == playerSymbol2))):
            w4 = w4 + 1

        # if 2 of Player-1's Symbol in a row with 1 open spots, above or below.
        if (((board[1] == ' ') and (board[4] == board[7] == playerSymbol1)) or
                ((board[7] == ' ') and (board[4] == board[1] == playerSymbol1)) or
                ((board[4] == ' ') and (board[1] == board[7] == playerSymbol1))):
            w3 = w3 + 1
        if (((board[2] == ' ') and (board[5] == board[8] == playerSymbol1)) or
                ((board[8] == ' ') and (board[2] == board[5] == playerSymbol1)) or
                ((board[5] == ' ') and (board[2] == board[8] == playerSymbol1))):
            w3 = w3 + 1
        if (((board[3] == ' ') and (board[6] == board[9] == playerSymbol1)) or
                ((board[9] == ' ') and (board[6] == board[3] == playerSymbol1)) or
                ((board[6] == ' ') and (board[3] == board[9] == playerSymbol1))):
            w3 = w3 + 1

        for i in range(1, 10, 3):  # iterating through each row on the board.
            # if 1 of Player-1's Symbol in a row with 2 open spots, right or left.
            if (((board[i] == playerSymbol1) and (board[i+1] == board[i+2] == ' ')) or
                    ((board[i+2] == playerSymbol1) and (board[i+1] == board[i] == ' ')) or
                    ((board[i+1] == playerSymbol1) and (board[i+2] == board[i] == ' '))):
                w1 = w1 + 1
            # if 1 of Player-2's Symbol in a row with 2 open spots, right or left.
            if (((board[i] == playerSymbol2) and (board[i+1] == board[i+2] == ' ')) or
                    ((board[i+2] == playerSymbol2) and (board[i+1] == board[i] == ' ')) or
                    ((board[i+1] == playerSymbol2) and (board[i+2] == board[i] == ' '))):
                w2 = w2 + 1
            # if 2 of Player-1's Symbols in a row and 1 open spot in same row
            if (((board[i] == board[i+1] == playerSymbol1) and (board[i+2] == ' ')) or
                    ((board[i+1] == board[i+2] == playerSymbol1) and (board[i] == ' ')) or
                    ((board[i+2] == board[i] == playerSymbol1) and (board[i+1] == ' '))):
                w3 = w3 + 1
            # if 2 of Player-2's Symbols in a row and 1 open spot in same row.
            if (((board[i] == board[i+1] == playerSymbol2) and (board[i+2] == ' ')) or
                    ((board[i+1] == board[i+2] == playerSymbol2) and (board[i] == ' ')) or
                    ((board[i+2] == board[i+1] == playerSymbol2) and (board[i+1] == ' '))):
                w4 = w4 + 1
            # if 3 of Player-1's Symbols in a row
            if board[i] == board[i+1] == board[i+2] == playerSymbol1:
                w5 = w5 + 1
            # if 3 of player-2's symbols in a row,
            if board[i] == board[i+1] == board[i+2] == playerSymbol2:
                w6 = w6 + 1
        # if 3 of player-2 symbols in a column.
        if (board[1] == board[4] == board[7] == playerSymbol2) or (board[3] == board[6] == board[9] == playerSymbol2) \
                or (board[2] == board[5] == board[8] == playerSymbol2):
            w6 = w6 + 1
        # if 3 of player-1's symbols in a diagonal
        if (board[1] == board[5] == board[9] == playerSymbol1) or (board[3] == board[5] == board[7] == playerSymbol1):
            w5 = w5 + 1
        # if 3 of Player-2's Symbol in a diagonal
        if (board[1] == board[5] == board[9] == playerSymbol2) or (board[3] == board[5] == board[7] == playerSymbol2):
            w6 = w6 + 1
        # if 3 of player-1 symbols in a column
        if (board[1] == board[4] == board[7] == playerSymbol1) or (board[3] == board[6] == board[9] == playerSymbol1) \
                or (board[2] == board[5] == board[8] == playerSymbol1):
            w5 = w5 + 1
        # if 2 of Player-1's Symbol in a diagonal
        if (((board[1] == ' ') and (board[5] == board[7] == playerSymbol1)) or
                ((board[7] == ' ') and (board[5] == board[1] == playerSymbol1)) or
                ((board[5] == ' ') and (board[1] == board[7] == playerSymbol1))):
            w7 = w7 + 1
        if (((board[9] == ' ') and (board[5] == board[3] == playerSymbol1)) or
                ((board[3] == ' ') and (board[9] == board[5] == playerSymbol1)) or
                ((board[5] == ' ') and (board[9] == board[3] == playerSymbol1))):
            w7 = w7 + 1
        # if 2 of Player-2's Symbol in a diagonal
        if (((board[1] == ' ') and (board[5] == board[7] == playerSymbol2)) or
                ((board[7] == ' ') and (board[5] == board[1] == playerSymbol2)) or
                ((board[5] == ' ') and (board[1] == board[7] == playerSymbol2))):
            w8 = w8 + 1
        if (((board[9] == ' ') and (board[5] == board[3] == playerSymbol2)) or
                ((board[3] == ' ') and (board[9] == board[5] == playerSymbol2)) or
                ((board[5] == ' ') and (board[9] == board[3] == playerSymbol2))):
            w8 = w8 + 1
        # Added 1 for bias term.
        feature_vector = [1, w1, w2, w3, w4, w5, w6, w7, w8]
        return feature_vector

    def boardPrint(self, board):
        """ Displays board in proper format"""
        print('   |   |')
        print(' ' + board[1] + ' | ' + board[2] + ' | ' + board[3])
        print('   |   |')
        print('-----------')
        print('   |   |')
        print(' ' + board[4] + ' | ' + board[5] + ' | ' + board[6])
        print('   |   |')
        print('-----------')
        print('   |   |')
        print(' ' + board[7] + ' | ' + board[8] + ' | ' + board[9])
        print('   |   |')

    def calculateNonFinalBoardScore(self, weight_vector, feature_vector):
        """
        Returns score/Value of a given non final board state. This is calculated as a linear regression.
        :params: weight_vector: vector of weights.
        :params: feature_vector: vector of features.
        :return: boardScore: returns the target variable.
        """
        weight_vector = np.array(weight_vector).reshape((len(weight_vector), 1))
        feature_vector = np.array(feature_vector).reshape((len(feature_vector), 1))
        boardScore = np.dot(weight_vector.T, feature_vector)  # Doing linear regression.
        return boardScore[0][0]  # Indexing to return a single value instead of np.array()

    def chooseMove(self, board, playerSymbol1, playerSymbol2):
        """
        Returns the best move from a set of legal moves for a given board state.
        :params: board: layout of a Tic-Tac-Toe board.
        :params: playerSymbol1: 'X' or "O' of one of the players. (self)
        :params: playerSymbol2: 'X" or 'O" of other player.
        """
        legalMoves, legalMovesIndex = self.look_for_legal_moves(board, playerSymbol1)
        # self.extractFeatures returns a feature vector X for a given board i.
        # iterating through all the boards that contains the next possible move and calculate the target variable for
        # each board.
        legalMoveScores = [self.calculateNonFinalBoardScore(self.playerTargetFunctionWeightVector,
                                                            self.extractFeatures(i, playerSymbol1, playerSymbol2)) for i
                           in legalMoves]
        # The move with the highest target variable is returned. The reason being is that the close we are to a win,
        # the higher the target variable.
        newBoard = legalMoves[np.argmax(legalMoveScores)]
        index = legalMovesIndex[np.argmax(legalMoveScores)]
        return newBoard, index

    def chooseRandomMove(self, board, playerSymbol):
        """ Returns a random move from a set of legal moves for a given board state """

        legalMoves, legalMovesIndex = self.look_for_legal_moves(board, playerSymbol)
        newBoard = random.choice(legalMoves)
        return newBoard


class PerformanceSystem:
    """
    Performance System takes the initial Game board & returns Solution trace/Game History of the Game Play.  This
    class basically simulates Tic-Tac-Toe game.
    """

    def __init__(self, initialBoard, playersTargetFunctionWeightVectors, playerSymbols):
        self.board = initialBoard
        # keeps track of each player's weight vector. Basically represents current V for both players.
        self.playersTargetFunctionWeightVectors = playersTargetFunctionWeightVectors
        self.player_symbols = playerSymbols  # ["X", "O"] or ["O", "X"] index 0 = player1, index 2 = player2

    def is_game_over(self, board, playerSymbol):
        """
        Returns True if game is over else returns false. Same function in Player class.
        """
        flag = False
        # Game already over
        if board == -1:
            flag = True
        # Game won by either player
        elif ((board[7] == board[8] == board[9] == playerSymbol) or  # Across the bottom.
              (board[4] == board[6] == board[5] == playerSymbol) or  # Across the middle.
              (board[1] == board[2] == board[3] == playerSymbol) or  # Across the top.
              (board[7] == board[4] == board[1] == playerSymbol) or  # Across the left side.
              (board[8] == board[5] == board[2] == playerSymbol) or  # Down the middle.
              (board[9] == board[6] == board[3] == playerSymbol) or  # Down the right side.
              (board[7] == board[5] == board[3] == playerSymbol) or  # Diagonal
              (board[9] == board[5] == board[1] == playerSymbol)):  # Diagonal
            flag = True
        # Board Full
        elif ' ' not in board[1:]:  # 0th index always left empty.
            flag = True
        return flag

    def generateGameHistory(self):
        """
        Returns Solution trace generated from pitting 2 players(agents) against each.
        """
        gameHistory = []
        gameStatusFlag = True
        player1 = Player(self.playersTargetFunctionWeightVectors[0], self.player_symbols[0])
        player2 = Player(self.playersTargetFunctionWeightVectors[1], self.player_symbols[1])
        tempBoard = copy.deepcopy(self.board)  # Current state of the game.
        while gameStatusFlag:
            # Loop until game is over.
            tempBoard, index = player1.chooseMove(tempBoard, player1.playerSymbol, player2.playerSymbol)
            gameHistory.append(tempBoard)
            gameStatusFlag = not self.is_game_over(tempBoard, player1.playerSymbol)  # True if game is over.
            if gameStatusFlag is False:  # if game is over, exit loop
                break
            # TODO: I'm not sure why player2 choose random moves. What happens if moves come from same target function?
            # I think its because the ai would not learn anything new. You can't improve if you don't see new data.
            tempBoard = player2.chooseRandomMove(tempBoard, player2.playerSymbol)
            # tempBoard = player2.chooseMove(tempBoard,player2.playerSymbol,player1.playerSymbol)
            gameHistory.append(tempBoard)
            gameStatusFlag = not self.is_game_over(tempBoard, player2.playerSymbol)
        return gameHistory


class Critic:
    """
    Critic takes the Game History & generates training examples to be used by Generalizer.
    """

    def __init__(self, gameHistory):
        self.gameHistory = gameHistory

    # Same function in Player class
    def extractFeatures(self, board, playerSymbol1, playerSymbol2):
        """
        Returns extracted feature Vector for a given board state where the features take on values from 0 to 3
        :params: board: layout of the board.
        :params: playerSymbol1: "X" or "O" representing the player. (self)
        :params: playerSymbol2: "X" or "O" representing the other player.
        :returns: feature_vector:
        """
        w1, w2, w3, w4, w5, w6, w7, w8 = 0, 0, 0, 0, 0, 0, 0, 0  # Initializing feature vector.
        # if 1 of Player-1's Symbol in a row with two open spots, above or below.
        if (((board[1] == playerSymbol1) and (board[4] == board[7] == ' ')) or
                ((board[7] == playerSymbol1) and (board[4] == board[1] == ' ')) or
                ((board[4] == playerSymbol1) and (board[1] == board[7] == ' '))):
            w1 = w1 + 1
        if (((board[2] == playerSymbol1) and (board[5] == board[8] == ' ')) or
                ((board[8] == playerSymbol1) and (board[2] == board[5] == ' ')) or
                ((board[5] == playerSymbol1) and (board[2] == board[8] == ' '))):
            w1 = w1 + 1
        if (((board[3] == playerSymbol1) and (board[6] == board[9] == ' ')) or
                ((board[9] == playerSymbol1) and (board[6] == board[3] == ' ')) or
                ((board[6] == playerSymbol1) and (board[3] == board[9] == ' '))):
            w1 = w1 + 1

        # if 1 of Player-2's Symbol in a row with 2 open spots, above or below.
        if (((board[1] == playerSymbol2) and (board[4] == board[7] == ' ')) or
                ((board[7] == playerSymbol2) and (board[4] == board[1] == ' ')) or
                ((board[4] == playerSymbol2) and (board[1] == board[7] == ' '))):
            w2 = w2 + 1
        if (((board[2] == playerSymbol2) and (board[5] == board[8] == ' ')) or
                ((board[8] == playerSymbol2) and (board[2] == board[5] == ' ')) or
                ((board[5] == playerSymbol2) and (board[2] == board[8] == ' '))):
            w2 = w2 + 1
        if (((board[3] == playerSymbol2) and (board[6] == board[9] == ' ')) or
                ((board[9] == playerSymbol2) and (board[6] == board[3] == ' ')) or
                ((board[6] == playerSymbol2) and (board[3] == board[9] == ' '))):
            w2 = w2 + 1

        # if 2 of Player-2's Symbol in a row with 1 open spots, above or below.
        if (((board[1] == ' ') and (board[4] == board[7] == playerSymbol2)) or
                ((board[7] == ' ') and (board[4] == board[1] == playerSymbol2)) or
                ((board[4] == ' ') and (board[1] == board[7] == playerSymbol2))):
            w4 = w4 + 1
        if (((board[2] == ' ') and (board[5] == board[8] == playerSymbol2)) or
                ((board[8] == ' ') and (board[2] == board[5] == playerSymbol2)) or
                ((board[5] == ' ') and (board[2] == board[8] == playerSymbol2))):
            w4 = w4 + 1
        if (((board[3] == ' ') and (board[6] == board[9] == playerSymbol2)) or
                ((board[9] == ' ') and (board[6] == board[3] == playerSymbol2)) or
                ((board[6] == ' ') and (board[3] == board[9] == playerSymbol2))):
            w4 = w4 + 1

        # if 2 of Player-1's Symbol in a row with 1 open spots, above or below.
        if (((board[1] == ' ') and (board[4] == board[7] == playerSymbol1)) or
                ((board[7] == ' ') and (board[4] == board[1] == playerSymbol1)) or
                ((board[4] == ' ') and (board[1] == board[7] == playerSymbol1))):
            w3 = w3 + 1
        if (((board[2] == ' ') and (board[5] == board[8] == playerSymbol1)) or
                ((board[8] == ' ') and (board[2] == board[5] == playerSymbol1)) or
                ((board[5] == ' ') and (board[2] == board[8] == playerSymbol1))):
            w3 = w3 + 1
        if (((board[3] == ' ') and (board[6] == board[9] == playerSymbol1)) or
                ((board[9] == ' ') and (board[6] == board[3] == playerSymbol1)) or
                ((board[6] == ' ') and (board[3] == board[9] == playerSymbol1))):
            w3 = w3 + 1

        for i in range(1, 10, 3):  # iterating through each row on the board.
            # if 1 of Player-1's Symbol in a row with 2 open spots, right or left.
            if (((board[i] == playerSymbol1) and (board[i+1] == board[i+2] == ' ')) or
                    ((board[i+2] == playerSymbol1) and (board[i+1] == board[i] == ' ')) or
                    ((board[i+1] == playerSymbol1) and (board[i+2] == board[i] == ' '))):
                w1 = w1 + 1
            # if 1 of Player-2's Symbol in a row with 2 open spots, right or left.
            if (((board[i] == playerSymbol2) and (board[i+1] == board[i+2] == ' ')) or
                    ((board[i+2] == playerSymbol2) and (board[i+1] == board[i] == ' ')) or
                    ((board[i+1] == playerSymbol2) and (board[i+2] == board[i] == ' '))):
                w2 = w2 + 1
            # if 2 of Player-1's Symbols in a row and 1 open spot in same row
            if (((board[i] == board[i+1] == playerSymbol1) and (board[i+2] == ' ')) or
                    ((board[i+1] == board[i+2] == playerSymbol1) and (board[i] == ' ')) or
                    ((board[i+2] == board[i] == playerSymbol1) and (board[i+1] == ' '))):
                w3 = w3 + 1
            # if 2 of Player-2's Symbols in a row and 1 open spot in same row.
            if (((board[i] == board[i+1] == playerSymbol2) and (board[i+2] == ' ')) or
                    ((board[i+1] == board[i+2] == playerSymbol2) and (board[i] == ' ')) or
                    ((board[i+2] == board[i+1] == playerSymbol2) and (board[i+1] == ' '))):
                w4 = w4 + 1
            # if 3 of Player-1's Symbols in a row
            if board[i] == board[i+1] == board[i+2] == playerSymbol1:
                w5 = w5 + 1
            # if 3 of player-2's symbols in a row,
            if board[i] == board[i+1] == board[i+2] == playerSymbol2:
                w6 = w6 + 1
        # if 3 of player-2 symbols in a column.
        if (board[1] == board[4] == board[7] == playerSymbol2) or (board[3] == board[6] == board[9] == playerSymbol2) \
                or (board[2] == board[5] == board[8] == playerSymbol2):
            w6 = w6 + 1
        # if 3 of player-1's symbols in a diagonal
        if (board[1] == board[5] == board[9] == playerSymbol1) or (board[3] == board[5] == board[7] == playerSymbol1):
            w5 = w5 + 1
        # if 3 of Player-2's Symbol in a diagonal
        if (board[1] == board[5] == board[9] == playerSymbol2) or (board[3] == board[5] == board[7] == playerSymbol2):
            w6 = w6 + 1
        # if 3 of player-1 symbols in a column
        if (board[1] == board[4] == board[7] == playerSymbol1) or (board[3] == board[6] == board[9] == playerSymbol1) \
                or (board[2] == board[5] == board[8] == playerSymbol1):
            w5 = w5 + 1
        # if 2 of Player-1's Symbol in a diagonal
        if (((board[1] == ' ') and (board[5] == board[7] == playerSymbol1)) or
                ((board[7] == ' ') and (board[5] == board[1] == playerSymbol1)) or
                ((board[5] == ' ') and (board[1] == board[7] == playerSymbol1))):
            w7 = w7 + 1
        if (((board[9] == ' ') and (board[5] == board[3] == playerSymbol1)) or
                ((board[3] == ' ') and (board[9] == board[5] == playerSymbol1)) or
                ((board[5] == ' ') and (board[9] == board[3] == playerSymbol1))):
            w7 = w7 + 1
        # if 2 of Player-2's Symbol in a diagonal
        if (((board[1] == ' ') and (board[5] == board[7] == playerSymbol2)) or
                ((board[7] == ' ') and (board[5] == board[1] == playerSymbol2)) or
                ((board[5] == ' ') and (board[1] == board[7] == playerSymbol2))):
            w8 = w8 + 1
        if (((board[9] == ' ') and (board[5] == board[3] == playerSymbol2)) or
                ((board[3] == ' ') and (board[9] == board[5] == playerSymbol2)) or
                ((board[5] == ' ') and (board[9] == board[3] == playerSymbol2))):
            w8 = w8 + 1
        # Added 1 for bias term.
        feature_vector = [1, w1, w2, w3, w4, w5, w6, w7, w8]
        return feature_vector

    def calculateNonFinalBoardScore(self, weight_vector, feature_vector):
        """
        Returns score/Value of a given non final board state. This is calculated as a linear regression.
        :params: weight_vector: vector of weights.
        :params: feature_vector: vector of features.
        :return: boardScore: returns the target variable.
        """
        weight_vector = np.array(weight_vector).reshape((len(weight_vector), 1))
        feature_vector = np.array(feature_vector).reshape((len(feature_vector), 1))
        boardScore = np.dot(weight_vector.T, feature_vector)  # Doing linear regression.
        return boardScore[0][0]  # Indexing to returning a single value instead of np.array().

    def calculateFinalBoardScore(self, board, playerSymbol1, playerSymbol2):
        """
        Returns score/Value of a given final board state
        """
        # If game ends in a draw
        score = 0
        # If player-1 (i.e self) wins
        if ((board[7] == board[8] == board[9] == playerSymbol1) or  # across the bottom
                (board[4] == board[5] == board[6] == playerSymbol1) or  # across the middle
                (board[1] == board[2] == board[3] == playerSymbol1) or  # across the top
                (board[7] == board[4] == board[1] == playerSymbol1) or  # down the left side
                (board[8] == board[5] == board[2] == playerSymbol1) or  # down the middle
                (board[9] == board[6] == board[3] == playerSymbol1) or  # down the right side
                (board[7] == board[5] == board[3] == playerSymbol1) or  # diagonal
                (board[9] == board[5] == board[1] == playerSymbol1)):  # diagonal
            score = 100
        # If player-2 (i.e opponent) wins
        elif((board[7] == board[8] == board[9] == playerSymbol2) or  # across the bottom
                (board[4] == board[5] == board[6] == playerSymbol2) or  # across the middle
                (board[1] == board[2] == board[3] == playerSymbol2) or  # across the top
                (board[7] == board[4] == board[1] == playerSymbol2) or  # down the left side
                (board[8] == board[5] == board[2] == playerSymbol2) or  # down the middle
                (board[9] == board[6] == board[3] == playerSymbol2) or  # down the right side
                (board[7] == board[5] == board[3] == playerSymbol2) or  # diagonal
                (board[9] == board[5] == board[1] == playerSymbol2)):  # diagonal
            score = -100
        return score

    def generateTrainingSamples(self, weight_vector, playerSymbol1, playerSymbol2):
        """
        Returns training examples i.e a list of list of feature vectors & associated scores.
        """
        trainingExamples = []
        # Iterate through every turn of the game except for first turn because board is empty.
        for i in range(len(self.gameHistory) - 1):
            feature_vector = self.extractFeatures(self.gameHistory[i + 1], playerSymbol1, playerSymbol2)
            # Append features and corresponding target variable.
            trainingExamples.append([feature_vector, self.calculateNonFinalBoardScore(weight_vector, feature_vector)])
        trainingExamples.append([self.extractFeatures(self.gameHistory[-1], playerSymbol1, playerSymbol2),
                                 self.calculateFinalBoardScore(self.gameHistory[-1], playerSymbol1, playerSymbol2)])
        return trainingExamples

    def arrayPrint(self, board):
        """
        Prints game board.
        """
        print('\n')
        print('   |   |')
        print(' ' + board[1] + ' | ' + board[2] + ' | ' + board[3])
        print('   |   |')
        print('-----------')
        print('   |   |')
        print(' ' + board[4] + ' | ' + board[5] + ' | ' + board[6])
        print('   |   |')
        print('-----------')
        print('   |   |')
        print(' ' + board[7] + ' | ' + board[8] + ' | ' + board[9])
        print('   |   |')
        print('\n')

    def boardDisplay(self, playerSymbol1, playerSymbol2, gameStatusCount):
        """
        Displays all moves & returns a list containing Win/Loss/Draw counts.
        :params: gameStatusCount: Holds current number of draws, wins and loses.
        :return: gameStatusCount: Return game count holder with recently played game added to the count.
        """
        for board in self.gameHistory:
            self.arrayPrint(board)
        finalScore = self.calculateFinalBoardScore(self.gameHistory[-1], playerSymbol1, playerSymbol2)
        # If game is a win
        if finalScore == 100:
            print(playerSymbol1 + " wins")
            gameStatusCount[0] = gameStatusCount[0] + 1
        # If game is a loss
        elif finalScore == -100:
            print(playerSymbol2 + " wins")
            gameStatusCount[1] = gameStatusCount[1] + 1
        # Else, game is a draw.
        else:
            print("Draw")
            gameStatusCount[2] = gameStatusCount[2] + 1
        return gameStatusCount


class Generalizer:
    """
    It takes Training examples from Critic & suggests/improves the Hypothesis function (Approximate Target Function).
    This object basically computed the updated weights for a linear model using gradient descent.
    """
    def __init__(self, trainingExamples):
        self.trainingExamples = trainingExamples

    def calculateNonFinalBoardScore(self, weight_vector, feature_vector):
        """ Returns score/Value of a given non final board state """
        weight_vector = np.array(weight_vector).reshape((len(weight_vector), 1))
        feature_vector = np.array(feature_vector).reshape((len(feature_vector), 1))
        boardScore = np.dot(weight_vector.T, feature_vector)
        return boardScore[0][0]

    def lmsWeightUpdate(self, weight_vector, alpha=0.4):
        """
        Returns new Weight vector updated y learning from Training examples via LMS (Least Mean Squares) training rule.
        trainingExamples are generated using current weight_vector.
        The implementation is equivalent to gradient descent with a mse loss function and linear model.
        The same weight_vector passed here is passed in Critic.generateTrainingSamples().
        Reference Deep learning by Goodfellow, Bengio, and Courville.
        """
        for trainingExample in self.trainingExamples:
            # This is the actual, y.
            vTrainBoardState = trainingExample[1]
            # This is the predicted value, $y_{hat}$.
            vHatBoardState = self.calculateNonFinalBoardScore(weight_vector, trainingExample[0])
            # This is the weight plus the a learning term multiplied by a term derived from the derivative.
            # Don't actually need full derivative term, only the direction.
            weight_vector = weight_vector + (alpha * (vTrainBoardState - vHatBoardState) * np.array(trainingExample[0]))
        return weight_vector


def train(numTrainingSamples=10):
    """
    Executions of Training & Testing phases. Player1 is the ai we're training.
    """
    # Training phase (Indirect Feedback via Computer v/s Computer)
    # Initializing variables.
    trainingGameCount = 0
    playerSymbols = ('X', 'O')  # index 1 is player1, index 2 is player2.
    playersTargetFunctionWeightVectors = [np.array([.5, .5, .5, .5, .5, .5, .5, .5, .5]),
                                          np.array([.5, .5, .5, .5, .5, .5, .5, .5, .5])]
    gameStatusCount = [0, 0, 0]  # Keeps track of number of wins, loses, and draws respectively.

    while trainingGameCount < numTrainingSamples:
        # Looping until generate numTrainingSamples games.
        experimentGenerator = ExperimentGenerator()
        # Creates empty Tic-Tac-Toe board.
        initialBoardState = experimentGenerator.generate_new_problem()

        performanceSystem = PerformanceSystem(initialBoardState, playersTargetFunctionWeightVectors, playerSymbols)
        # Plays a single game of Tic-Tac-Toe.
        gameHistory = performanceSystem.generateGameHistory()

        critic = Critic(gameHistory)
        # Generate training examples from the game history.
        # Each board in the game history will be used to generate a training instance.
        trainingExamplesPlayer1 = critic.generateTrainingSamples(playersTargetFunctionWeightVectors[0],
                                                                 playerSymbols[0], playerSymbols[1])
        # Getting the number of wins, loses, draws.
        gameStatusCount = critic.boardDisplay(playerSymbols[0], playerSymbols[1], gameStatusCount)

        generalizer = Generalizer(trainingExamplesPlayer1)
        playersTargetFunctionWeightVectors = [generalizer.lmsWeightUpdate(playersTargetFunctionWeightVectors[0]),
                                              generalizer.lmsWeightUpdate(playersTargetFunctionWeightVectors[1])]
        trainingGameCount = trainingGameCount + 1

    # Epoch wise Game status
    print("\nTraining Results: (" + "Player-1 Wins = " + str(gameStatusCount[0]) +
          ", Player-2 Wins = " + str(gameStatusCount[1]) + ", Game Draws = " + str(gameStatusCount[2]) +
          ")\n")

    # Getting final weights.
    learntWeight = list(np.mean(np.array([playersTargetFunctionWeightVectors[0],
                                          playersTargetFunctionWeightVectors[1]]), axis=0))
    print("Final Learnt Weight Vector: \n" + str(learntWeight))

    return learntWeight


def run(weights):
    """
    Player is always O and computer always X.
    :param weights:
    :return:
    """
    # Computer vs Human Games
    print("\nDo you want to play(y/n) v/s Computer AI")
    ans = input()
    while ans == "y":

        experimentGenerator = ExperimentGenerator()
        boardState = experimentGenerator.generate_new_problem()
        gameStatusFlag = True
        computer = Player(weights, 'X')  # Pass learned weights to the new AI.
        gameHistory = []

        print('\nBegin Computer(X) v/s Human(O) Tic-Tac-Toe\n')
        while gameStatusFlag:

            boardState, index = computer.chooseMove(boardState, computer.playerSymbol, 'O')
            print('Computers\'s Turn:\n')
            computer.boardPrint(boardState)
            gameHistory.append(boardState)
            gameStatusFlag = not computer.is_game_over(boardState, computer.playerSymbol)
            if gameStatusFlag is False:
                break

            print('Human\'s Turn:\n')
            print('Enter position  (1-9):')
            x = int(input())

            boardState[x] = 'O'
            computer.boardPrint(boardState)
            gameHistory.append(boardState)
            gameStatusFlag = not computer.is_game_over(boardState, 'O')

        print("Do you want to continue playing(y/n).")
        ans = input()
        if ans != 'y':
            break



def main():
    learned_weights = train(10000)
    run(learned_weights)


if __name__ == '__main__':
   main()
