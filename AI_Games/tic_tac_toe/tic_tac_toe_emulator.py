import random
import linear_regression_ai


class TicTacToe(object):
    def __init__(self, ai=None):
        """
        Instantiates the game.
        :param ai: Object representing the trained agent. Agent must have a method called chooseMove() that returns an
        index representing the next move to make.
        """
        self.player = None
        self.computer = None
        self.first = None
        self.board = [' '] * 10  # index 0 is not used.
        self.ai = ai

    def draw_board(self):
        """
        This function prints out the board that it was passed.  Note index 0 is not used.
        """
        board = self.board
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

    def input_player_letter(self):
        """
        Lets the player type which letter they want to be.
        :return: a list with the player's letter as the first item, and the computer's letter as the second.
        """
        letter = ''
        while not (letter == 'X' or letter == 'O'):
            # Continuously ask player if they want to be X or O.
            print("Do you want to be X or O?")
            letter = input().upper()

        if letter == "X":
            # The first element in the list is the player's letter, second is the computer's letter.
            self.player = 'X'
            self.computer = 'O'
        else:
            self.player = 'O'
            self.computer = 'X'

    def who_goes_first(self):
        """
        Randomly choose the player who goes first.
        :return:
        """
        if random.randint(0, 1) == 0:
            self.first = 'computer'
            return 'computer'
        else:
            self.first = 'player'
            return 'player'

    def make_move(self, letter, pos):
        """
        Function that does a move on the board.
        :param letter:  What letter to add onto the board.
        :param pos:  Position to add onto board.
        :return:
        """
        self.board[pos] = letter

    @staticmethod
    def static_make_move(copy_board, letter, pos):
        copy_board[pos] = letter
        return copy_board

    def is_winner(self, person):
        """
        Determines if the last move made is a winning move. Return True if it is, else False.
        :return: True if player has won.
        """
        if person == 'player':
            return ((self.board[7] == self.player and self.board[8] == self.player and self.board[
                9] == self.player) or  # across the bottom
                    (self.board[4] == self.player and self.board[5] == self.player and self.board[
                        6] == self.player) or  # across the middle
                    (self.board[1] == self.player and self.board[2] == self.player and self.board[
                        3] == self.player) or  # across the top
                    (self.board[7] == self.player and self.board[4] == self.player and self.board[
                        1] == self.player) or  # down the left side
                    (self.board[8] == self.player and self.board[5] == self.player and self.board[
                        2] == self.player) or  # down the middle
                    (self.board[9] == self.player and self.board[6] == self.player and self.board[
                        3] == self.player) or  # down the right side
                    (self.board[7] == self.player and self.board[5] == self.player and self.board[
                        3] == self.player) or  # diagonal
                    (self.board[9] == self.player and self.board[5] == self.player and self.board[
                        1] == self.player))  # diagonal
        else:
            return ((self.board[7] == self.computer and self.board[8] == self.computer and self.board[
                9] == self.computer) or  # across the top
                    (self.board[4] == self.computer and self.board[5] == self.computer and self.board[
                        6] == self.computer) or  # across the middle
                    (self.board[1] == self.computer and self.board[2] == self.computer and self.board[
                        3] == self.computer) or  # across the bottom
                    (self.board[7] == self.computer and self.board[4] == self.computer and self.board[
                        1] == self.computer) or  # down the left side
                    (self.board[8] == self.computer and self.board[5] == self.computer and self.board[
                        2] == self.computer) or  # down the middle
                    (self.board[9] == self.computer and self.board[6] == self.computer and self.board[
                        3] == self.computer) or  # down the right side
                    (self.board[7] == self.computer and self.board[5] == self.computer and self.board[
                        3] == self.computer) or  # diagonal
                    (self.board[9] == self.computer and self.board[5] == self.computer and self.board[
                        1] == self.computer))  # diagonal

    @staticmethod
    def static_is_winner(bo, le):
        """
        Given a board and a player’s letter, this function returns True if that player has won.
        We use bo instead of board and le instead of letter so we don’t have to type as much.
        :params: bo: this is the board
        :params: le: this is the move
        :return: True or False.
        """
        return ((bo[7] == le and bo[8] == le and bo[9] == le) or  # across the bottom
                (bo[4] == le and bo[5] == le and bo[6] == le) or  # across the middle
                (bo[1] == le and bo[2] == le and bo[3] == le) or  # across the top
                (bo[7] == le and bo[4] == le and bo[1] == le) or  # down the left side
                (bo[8] == le and bo[5] == le and bo[2] == le) or  # down the middle
                (bo[9] == le and bo[6] == le and bo[3] == le) or  # down the right side
                (bo[7] == le and bo[5] == le and bo[3] == le) or  # diagonal
                (bo[9] == le and bo[5] == le and bo[1] == le))  # diagonal

    def get_board_copy(self):
        """
        Make a duplicate of the board list and return the duplicate.
        :return: dupeBoard: replica of the board.
        """
        dupeBoard = []

        for i in self.board:  # for element in board
            dupeBoard.append(i)
        return dupeBoard

    def is_space_free(self, move):
        """
        Return True if the passed move is free on the board.
        :param move: The position player or computer wants to place onto board.
        :return: True or False.
        """
        return self.board[move] == ' '  # Return True of empty, else return False.

    def get_player_move(self):
        """
        Lets the player type in their move.
        :return: index of the move
        """
        move = " "

        # if move is not 1 to 9 or if space is filled.
        while move not in '1 2 3 4 5 6 7 8 9'.split() or not self.is_space_free(int(move)):
            print("What is your next move? (1-9)")
            move = input()

        self.make_move(self.player, int(move))  # Making player's move.

    def choose_random_move_from_list(self, moves_list):
        """
        Returns a valid move from the passed list on the board.
        :param moves_list: A list of valid moves.
        :return: Valid move or return None
        """
        possibleMoves = []
        for i in moves_list:  # For move in moveList
            if self.is_space_free(i):
                # if possible move, append to list
                possibleMoves.append(i)
        if len(possibleMoves) != 0:
            return random.choice(possibleMoves)
        else:
            return None

    def get_computer_move(self):
        """
        Given a board and the computer's letter, determine where to move and return that move. This is the computer
        AI implementation.
        :return: Returns the next move.
        """
        copy = self.get_board_copy()
        if self.ai is None:  # if did not pass in a trained agent.
            # First, check if computer can win in the next move
            for i in range(1, 10):
                if self.is_space_free(i):  # If space is free
                    copy = TicTacToe.static_make_move(copy, self.computer, i)  # make move
                    if TicTacToe.static_is_winner(copy, self.computer):
                        return i

            # if the player could win in next move, then block them.
            for i in range(1, 10):
                copy = self.get_board_copy()
                if self.is_space_free(i):  # If space is free
                    copy = TicTacToe.static_make_move(copy, self.player, i)  # make move
                    if TicTacToe.static_is_winner(copy, self.player):
                        return i

            # Try to take one of the corners, if they are free by passing integer corners.
            move = self.choose_random_move_from_list([1, 3, 7, 9])
            if move is not None:
                return move

            # Try to take center, if it is open.
            if self.is_space_free(5):
                return 5

            # place move on the middle space of the sides. Worst move to make in the beginning.
            return self.choose_random_move_from_list([2, 4, 6, 8])
        else:
            nextBoard, nextMove = self.ai.chooseMove(copy, self.computer, self.player)
            return nextMove

    def is_board_full(self):
        """
        Return True if every space on the board has been taken. Otherwise, return False.
        :return:
        """
        for i in range(1, 10):
            if self.is_space_free(i):
                return False
        return True

    def run(self):
        print("Welcome to Tic Tac  Toe!")
        self.input_player_letter()
        turn = self.who_goes_first()  # Determining who goes first. player or computer.
        print("The " + turn + " will go first.")

        while True:
            if turn == 'player':
                # if player's turn.
                self.draw_board()
                self.get_player_move()

                if self.is_winner(turn):  # Determine if the move made is winning move.
                    self.draw_board()
                    print("Hooray! You have won the game!")
                    break
                else:
                    if self.is_board_full():
                        self.draw_board()
                        print("The game is a tie!")
                        break
                    else:
                        turn = 'computer'
            else:
                # Computer's turn.
                move = self.get_computer_move()
                self.make_move(self.computer, move)

                if self.is_winner(turn):
                    self.draw_board()
                    print("The computer has beaten you! You lose.")
                    break
                else:
                    if self.is_board_full():
                        self.draw_board()
                        print("The game is a tie!")
                        break
                    else:
                        turn = 'player'


def play_again():
    """
    This function returns True if the player wants to play again, otherwise return False.
    :return: temp: True or False.
    """
    print("Do you want to play again? (yes or no)")
    temp = input().lower().startswith('y')  # True if starts with y.
    return temp


def main():
    weights = linear_regression_ai.train(10000)
    agent = linear_regression_ai.Player(playerTargetFunctionWeightVector=weights)
    while True:
        engine = TicTacToe(ai=agent)
        engine.run()
        play = play_again()
        if not play:  # if play again is false
            break


if __name__ == '__main__':
    main()
