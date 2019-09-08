# Problem 7
# we can assume length of list is even
def mSort(array):
    # Base case.
    if len(array) == 1:
        return array
    # Recursive step
    mid = len(array)//2
    fh = mSort(array[0:mid])
    lh = mSort(array[mid:])
    return (merge(fh, lh))

def merge(s1, s2):
    if s1 == []:
        return s2
    elif s2 == []:
        return s1
    elif s1[0] <= s2[0]:
        temp = s1.pop(0)
    elif s1[0] >= s2[0]:
        temp = s2.pop(0)
    newA = merge(s1, s2)
    newA.insert(0, temp)
    return newA

def main():
    userInput = input("Please enter a sequence of numbers separated by a ',':")
    userList = [int(x) for x in userInput.split(",")]
    result = mSort(userList)
    print(result)

if __name__ == '__main__':
    main()



