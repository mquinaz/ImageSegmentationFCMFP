from itertools import permutations
perms = [''.join(p) for p in permutations('0123456')]
output = []


matrix_size = 6
total = 0
matrixA = ['0','2','1','3','4','5','6']
matrixB = [0,6,1,5,4,2,2]
for P in perms:
	correct = 0
	for pos in range(0,matrix_size):
		if matrixA[pos] == P[int(matrixA[pos])]:
			correct += 1
	output.append(correct)

print output

