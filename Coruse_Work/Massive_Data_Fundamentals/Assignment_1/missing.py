#!/usr/bin/env python
import sys

def unique(alist):
    uniqs = []
    for num in alist:
        if num not in uniqs:
            uniqs.append(num)
    return uniqs

def missing(N, nl):
    try:
        N = int(N)
    except:
        print("a non-integer value is provided to the 1st argument")
        sys.exit(1) # This is to prevent a traceback from being shown.
    if len(nl) != N-1:
       print("The number of items provided in the second argument does not equal", N-1)
    elif len(unique(nl)) < len(nl):
        print("There are duplicate values in the second argument")
    else: 
        try:
            nums = [int(x) for x in nl]
        except:
            print("a non-integer value is provided to the 2nd argument")
            sys.exit(1)
        ref = [i for i in range(1, N+1)]
        for a in ref:
            if a not in nums:
                print("The missing number is", a)

def main():
    # Let the 1st argument be an string n where n is the max number in the 2nd argument.
    # Let the 2nd argument be a string of consequtive integers separated by a space. 
    # The 2nd argument has n-1 integers.
    
    if len(sys.argv) == 3:
        n = sys.argv[1]
        numString = sys.argv[2]
        numList = numString.split(" ") 
        missing(n, numList) 
    else:
        print("usage:", sys.argv[0], "n", "String of integers separated by a single space, number of integers is n-1, and no integer is larger than n")

if __name__ == "__main__":
    main()
