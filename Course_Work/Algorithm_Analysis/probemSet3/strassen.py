from optparse import OptionParser
import time
import sys


def init_matrix(filename):
    entries = open(filename, 'r').read().split('\n')
    entries = [int(entry) for entry in entries]
    mid = dim**2 # mid point seprates entries from matrix a and b.
    matrix_a = [entries[i:i+dim] for i in range(0, len(entries[0:mid]), dim)]
    matrix_b = [entries[i:i+dim] for i in range(mid, len(entries), dim)]
    return matrix_a, matrix_b


def conven_matrix_multi(A,B): 
    # The cross 
    # List to store matrix multiplication result 
    d = get_dim(A)[0]
    ret = [[0 for j in range(0, d)] for i in range(0, d)]
    
    for i in range(0, d):  
        for j in range(0, d): 
            for k in range(0, d):  
                ret[i][j] += A[i][k] * B[k][j]  
    return ret

# Call matrix_multiplication function 
#conven_matrix_multi(M,N) 

def matrix_addition(matrix_a, matrix_b):
    # conventional matrix addition.  
#    matrix = []
#    for row in range(len(matrix_a)):
#        entry = []
#        for col in range(len(matrix_a[row])):
#            entry.append(matrix_a[row][col] + matrix_b[row][col])
#        matrix.append(entry)
#    return matrix
    
    return [[matrix_a[i][j] + matrix_b[i][j]
             for j in range(len(matrix_a[i]))] 
                for i in range(len(matrix_a))]

    
def matrix_subtraction(matrix_a, matrix_b):
    # conventional matrix subtraction.
#    matrix = []
#    for row in range(len(matrix_a)):
#        entry = []
#        for col in range(len(matrix_a[row])):
#            entry.append(matrix_a[row][col] - matrix_b[row][col])
#        matrix.append(entry)
#    return matrix
    
    return [[matrix_a[i][j] - matrix_b[i][j]
             for j in range(len(matrix_a[i]))] 
                for i in range(len(matrix_a))]


def split_matrix(A):
    d = get_dim(A)[0] # number of rows or number of columns.
    mid = d // 2
    # ith row, jth column.  
    top_left = [[A[i][j] for j in range(mid)] for i in range(mid)]
    top_right = [[A[i][j] for j in range(mid, d)] 
            for i in range(mid)]
    bot_left = [[A[i][j] for j in range(mid)] 
                for i in range(mid, d)]
    bot_right = [[A[i][j] for j in range(mid, d)] 
                for i in range(mid, d)]
    return top_left, top_right, bot_left, bot_right


def get_dim(matrix):
    return len(matrix), len(matrix[0])

def strassen(matrix_a, matrix_b):
    """
    Strassen Algorithm implementation.  
    Only works for matrices of even length (2x2)
    """    
    # switch to conventional algo when n = cross_point.  
    d = get_dim(matrix_a)
    if d <= (cross_point, cross_point):
        return conven_matrix_multi(matrix_a, matrix_b)
    if d[0] % 2 != 0:
        matrix_a, matrix_b = padding(matrix_a, matrix_b)
    
    A, B, C, D = split_matrix(matrix_a)
    E, F, G, H = split_matrix(matrix_b)

    p1 = strassen(A, matrix_subtraction(F, H))
    p2 = strassen(matrix_addition(A, B), H)
    p3 = strassen(matrix_addition(C, D), E)
    p4 = strassen(D, matrix_subtraction(G, E))
    p5 = strassen(matrix_addition(A, D), matrix_addition(E, H))
    p6 = strassen(matrix_subtraction(B, D), matrix_addition(G, H))
    p7 = strassen(matrix_subtraction(A, C), matrix_addition(E, F))

    top_left = matrix_addition(matrix_subtraction(
            matrix_addition(p5, p4), p2), p6)
    top_right = matrix_addition(p1, p2)
    bot_left = matrix_addition(p3, p4)
    bot_right = matrix_subtraction(matrix_subtraction(
            matrix_addition(p1, p5), p3), p7)

    # construct new matrix from the 4 sub-matrices.  
    fin_matrix = []
    for i in range(len(top_left)):
        fin_matrix.append(top_left[i] + top_right[i])
    for i in range(len(bot_right)):
        fin_matrix.append(bot_left[i] + bot_right[i])
    return fin_matrix

def padding(matrix_a, matrix_b):
    ''' 
    adding 0's to convert dimensions of matrix into power of 2
    '''
    d = get_dim(matrix_a)[0]
    new_dim = d + 1
    A = [[0 for j in range(new_dim)] for i in range(new_dim)]
    B = [[0 for j in range(new_dim)] for i in range(new_dim)]
    
    for i in range(d):
        for j in range(d):
            A[i][j] = matrix_a[i][j]
            B[i][j] = matrix_b[i][j]
    return A, B

def stdout(matrix_a):
    for i in range(get_dim(matrix_a)[0]):
        print('\n', matrix_a[i][i])
    
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-f", dest='input_filename', default='input.txt',
                      help='Specify the input file that contains 2 matrices\
                      where every entry is separated by \n.  Both matrices \
                      are required to be even length.', 
                      action='store', type='string')
    parser.add_option("-c", dest='crossover_size', default=26,
                      help='The size of the matrix when conventional algo\
                      should be used instead of strassen', action='store',
                      type='int')
    parser.add_option("-d", dest='dimension', default=6,
                      help='The length of the matrix.', action='store',
                      type='int')
    parser.add_option("-t", dest='time', default=0,
                      help="1 = return time, 0 = no time", action='store',
                      type='int')
    parser.add_option("-o", dest='output', default=0,
                      help="1 = show final output, 0 = don't show output", 
                      action='store', type='int')
    (options, args) = parser.parse_args()
    
    cross_point = options.crossover_size
#    mat_filename = options.input_filename
    mat_filename = str(sys.argv[3])
#    dim = options.dimension    
    dim = int(sys.argv[2])
    timer = options.time
    out = options.output
    
    
    matrix_a, matrix_b = init_matrix(mat_filename)
    start = time.time()
    product = strassen(matrix_a, matrix_b)
    end = time.time()
    stdout(product)
    
    if timer == 1:
        print("\nRuntime in seconds is:", end-start)













