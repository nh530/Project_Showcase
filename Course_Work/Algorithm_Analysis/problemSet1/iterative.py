import time

def fib(n):
    # n is the nth fibonacci number
    Array = [0,1]
    for i in range(2, n+1):
        Array.append((Array[i-1] + Array[i-2]))
    return Array[n]

def fibMod(n):
    '''
    This function returns the nth Fibonacci number modulo 65536 by using the 
    fact that the nth fibonacci number is the sum of n-1 fibonacci number and 
    n-2 fibonacci number.  
    '''
    Array = [0,1]
    # range(2, n+1) represents the range of numbers from 2 to n.
    for i in range(2, n+1):
        # The ith fibonacci number is the sum of the last 2 fibonacci numbers.
        Array.append((Array[i-1]%65536 + Array[i-2]%65536)%65536)
    return Array[n]

def main():
    # running the fibMod function for ith fib number modulo 65536 for 1 minute 
    # The highest Fibonacci number reached is 15078.
    endTime = time.time() + 60
    i = 0
    while time.time() < endTime:
        fibMod(i)
        if time.time() >= endTime:
            print("The", i, "fibonacci number modulo 65536 is:", fibMod(i))
        i = i + 1

if __name__ == '__main__':
    main()
    



