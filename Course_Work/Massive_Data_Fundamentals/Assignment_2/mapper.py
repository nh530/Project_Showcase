#!/usr/bin/env python
import sys
import re

def convMonth(month):
    converter = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06', 
                'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}
    num = converter[month]
    return num

def parser(text):
    # text is a si ngle string
    # prints to stdout a key \t value string where key is the month
    # and value is the frequency, which is 1.
    pat1 = re.compile(r"[0-9]+/[A-Za-z]+/[0-9]+:[0-9]+:[0-9]+:[0-9]")
    matches = pat1.findall(text)
    post1 = [text.split(":")[0] for text in matches]
    for date in  post1:
        day, mon, year = date.split("/")
        monNum = convMonth(mon)
        out = year + "-" + monNum
        sys.stdout.write("{}\t1\n".format(out))

def main():
    for line in sys.stdin:
        parser(line)

if __name__ == '__main__':
    main()
