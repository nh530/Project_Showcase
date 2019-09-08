import time

def fib(n):
    # n is the nth fibonacci number
    # Assume n is 2^i for i>0  
    Array = [[0,1], [1,1]]
    i = 1
    # The purpose of this while loop is to determine i.
    while True:
        if n == (2**i):
            break
        i = i + 1
    for j in range(1, i+1):
        tl = (Array[0][0] * Array[0][0]) + (Array[0][1] * Array[1][0])
        tr = (Array[0][0] * Array[0][1]) + (Array[0][1] * Array[1][1])
        bl = (Array[1][0] * Array[0][0]) + (Array[1][1] * Array[1][0])
        br = (Array[1][0] * Array[0][1]) + (Array[1][1] * Array[1][1])
        Array = [[tl, tr], [bl, br]]
    return tr

def fibMod(n):
    '''
    Returns the nth Fibonacci number modulo 65536 by using the matrix 
    multiplication method.  
    '''
    # n is the nth fibonacci number.
    # Assume n is 2^i for i>0.
    Array = [[0,1], [1,1]]
    i = 1
    # The purpose of this while loop is to determine the value of i.
    while True:
        if n == (2**i):
            break
        i = i + 1
    # This for loop is used to compute i+1 matrix multiplications for 2x2
    # matrix.  
    for j in range(1, i+1):
        tl = ((Array[0][0] * Array[0][0])%65536 + (Array[0][1] * Array[1][0]
        )%65536)%65536
        tr = ((Array[0][0] * Array[0][1])%65536 + (Array[0][1] * Array[1][1]
        )%65536)%65536
        bl = ((Array[1][0] * Array[0][0])%65536 + (Array[1][1] * Array[1][0]
        )%65536)%65536
        br = ((Array[1][0] * Array[0][1])%65536 + (Array[1][1] * Array[1][1]
        )%65536)%65536
        # tl is the top left of matrix, tr is top right, bl is botom left,
        # br is bottom right.
        Array = [[tl, tr], [bl, br]]
    return tr

def main():
    # running the fibMod function for ith fib number modulo 65536 for 1 minute 
    endTime = time.time() + 60
    i = 1
    while time.time() < endTime:
        fibMod(2**i)
        if time.time() >= endTime:
            print("The", 2**i, "fibonacci number modulo 65536 is:", 
                  fibMod(2**i))
        i = i + 1

if __name__ == '__main__':
    main()


