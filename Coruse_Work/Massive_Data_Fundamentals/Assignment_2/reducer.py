#!/usr/bin/env python
import sys
    
    
def main():
    curkey = None
    total = 0
    collection = {}
    for line in sys.stdin:
        key, val = line.split("\t")
        if key not in collection:
            collection[key] = 1
        else:
            collection[key] = collection[key] + int(val)
    for key in collection: 
        sys.stdout.write("{},{}\n".format(key, collection[key]))
if __name__ == '__main__':
    main()
    
