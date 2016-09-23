import sys

if not len(sys.argv) == 3:
	print('usage: python3 %s <index> <file name>' % sys.argv[0])
	exit()

idx = int(sys.argv[1])
f = sys.argv[2]

num = [[float(line.strip().split()[idx]), line.strip().split()[idx]] for line in open(f)]
sorted_num = [i for i in sorted(num, key=lambda x: x[0])]

with open('ans1.txt', 'w') as fw:
	fw.write(','.join([i[1] for i in sorted_num]) + '\n')
