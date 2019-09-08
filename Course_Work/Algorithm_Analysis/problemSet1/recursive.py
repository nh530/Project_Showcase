import time

def fib(n):
    # n is the nth fibonacci number
    if n == 0:
        return 0
    if n == 1:
        return 1
    else:
        return (fib(n-1) + fib(n-2))

def fibMod(n):
    # n is the nth fibonacci number
    '''
    This function returns 0 if n is 0 and 1 if n is 1.  If n is greater than
    1, then this function computes the nth fibonacci number modulo 65536by recusively 
    calling on n-1 and n-2 until it reaches the base cases, n = 0 and n = 1.  
    '''
    if n == 0:
        return 0
    if n == 1:
        return 1
    else:
        return (fibMod(n-1)%65536 + fibMod(n-2)%65536)%65536

def main():
    # running the fibMod function for ith fib number modulo 65536 for 1 minute 
    # The highest Fiibonacci number reached is 38.  
    endTime = time.time() + 60
    i = 0
    while time.time() < endTime:
        fibMod(i)
        if time.time() >= endTime:
            print("The", i, "fibonacci number modulo 65536 is:", fibMod(i))
        i = i + 1

if __name__ == '__main__':
    main()






