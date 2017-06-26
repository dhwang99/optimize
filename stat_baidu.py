#encoding: utf8

import sys
ff = open(sys.argv[1],'r')

for line in ff:
    arr = line.strip().split('\t')
    if len(arr) != 7:
        continue

    upath, ctime2, ed2, etime2, ctime1, ed1,etime1 = arr

