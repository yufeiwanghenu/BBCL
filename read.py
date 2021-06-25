import os
name = []
file = open(r'E:\wyf\DOTAdata\test.txt','w')
for i, j, k in os.walk(r'E:\wyf\DOTAdata\images'):
    name = k[:]

for u in name:
    file.write(u[:-4] + '\n')