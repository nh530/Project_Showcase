#!/usr/bin/python

import sys
import re

def extract_names(filename):
  """
  Given a file name for baby<year>.html,
  returns a list starting with the year string
  followed by the name-rank strings in alphabetical order.
  
  ['2006', 'Aaliyah 91', Aaron 57', 'Abagail 895', ' ...]
  """
  with open(filename, 'r', encoding='utf-8') as f:
    text = f.read() 
    # text should be a string of the entire html code
  pattern = re.compile(r'<td>[0-9]+</td><td>[A-Z][a-z]+</td>' \
            '<td>[A-Z][a-z]+</td>')
  matches = pattern.findall(text) 
  # matches is a list of rank and name 
  post1 = [name.split('</td><td>') for name in matches] 
  # post1 is a list of lists where each nested list contains 
  # boy and girl names of the same rank.
  post2 = {}
  for lists in post1:
    temp = []
    for element in lists:
      temp.append(element.strip("<td>").strip("</td>").strip())
    post2[temp[1]] = temp[0]
    post2[temp[2]] = temp[0]
  ranks = [(key + " " + value) for key, value in post2.items()]
  year = filename[14:18]
  sortL = sorted(ranks)
  sortL.insert(0, year)
  return sortL

def main():
  # This command-line parsing code is provided.
  # Make a list of command line arguments, omitting the [0] element
  # which is the script itself.

  if len(sys.argv) == 2:
    arg = sys.argv[1] # arg is the filename.
  else:
    print("usage: ", sys.argv[0], "filename")
    sys.exit(1)
  path = 'babynames/' + arg
  # For each filename, get the names, then print the text output
  names = extract_names(path)
  for element in names:
    print(element)

if __name__ == '__main__':
  main()
