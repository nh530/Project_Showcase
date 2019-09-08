#!/usr/bin/env python
import fileinput
import re

def getWords(text):
    '''
    This function turns a list of strings into a single string of uncleaned
    words that is separated by a space.  
    Parameters:
    text (String): A list of strings to be parsed.
    Return:
    string (String): A single string of uncleaned words separated by a space.
    '''
    lines = [lines.split(" ") for lines in text]
    words = [word.strip().lower() for elements in lines for word in elements]
    string = ""
    for word in words:
        string = (string + " " + word)        
    return string

def cleaner(text):
    ''' 
    This function find all words in a string, sorts the words in alphabetical
    order, and prints out each word 1 by 1.  This function also cleans each
    word by removing unnecessary symbols and non-alphanumeric characters. 

    Parameters:
    text (String): A string that will be parsed for words. 
    Return:
    None
    '''
    pat1 = re.compile(r"[a-z]+-[a-z]+")
    mat1 = pat1.findall(text)
    pat2 = re.compile(r"[a-z]+")
    mat2 = pat2.findall(text)
    uniqs = set()
    mat1.extend(mat2)
    for word in mat1:
        uniqs.add(word)
    sortW = sorted(uniqs)
    for word in sortW:
        print(word)

def main():
    text = fileinput.input()
    collection = [lines for lines in text]
    newC = getWords(collection)
    cleaner(newC)

if __name__ == '__main__':
    main() 
