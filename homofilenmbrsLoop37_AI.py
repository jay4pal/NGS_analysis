import numpy as np

a = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
b = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
c = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
'''
a = np.array([0, 1, 2, 3, 4, 5, 6,7,8,9])
b = np.array([0, 1, 2, 3, 4, 5, 6,7,8,9])
c = np.array([0, 1, 2, 3, 4, 5, 6,7,8,9])
'''
### BLOSUM62 substitution matrix -> Penalty for X --> -5; lowest mismatch score -4
'''s= np.array([[-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5],
             [-5, 9,-1,-1,-3, 0,-3,-3,-3,-4,-3,-3,-3,-3,-1,-1,-1,-1,-2,-2,-2],
             [-5,-1, 4, 1,-1, 1, 0, 1, 0, 0, 0,-1,-1, 0,-1,-2,-2,-2,-2,-2,-3],
             [-5,-1, 1, 5,-1, 0,-2, 0,-1,-1,-1,-2,-1,-1,-1,-1,-1, 0,-2,-2,-2],
             [-5,-3,-1,-1, 7,-1,-2,-2,-1,-1,-1,-2,-2,-1,-2,-3,-3,-2,-4,-3,-4],
             [-5, 0, 1, 0,-1, 4, 0,-2,-2,-1,-1,-2,-1,-1,-1,-1,-1, 0,-2,-2,-3],
             [-5,-3, 0,-2,-2, 0, 6, 0,-1,-2,-2,-2,-2,-2,-3,-4,-4,-3,-3,-3,-2],
             [-5,-3, 1, 0,-2,-2, 0, 6, 1, 0, 0, 1, 0, 0,-2,-3,-3,-3,-3,-2,-4],
             [-5,-3, 0,-1,-1,-2,-1, 1, 6, 2, 0,-1,-2,-1,-3,-3,-4,-3,-3,-3,-4],
             [-5,-4, 0,-1,-1,-1,-2, 0, 2, 5, 2, 0, 0, 1,-2,-3,-3,-2,-3,-2,-3],
             [-5,-3, 0,-1,-1,-1,-2, 0, 0, 2, 5, 0, 1, 1, 0,-3,-2,-2,-3,-1,-2],
             [-5,-3,-1,-2,-2,-2,-2, 1,-1, 0, 0, 8, 0,-1,-2,-3,-3,-3,-1, 2,-2],
             [-5,-3,-1,-1,-2,-1,-2, 0,-2, 0, 1, 0, 5, 2,-1,-3,-2,-3,-3,-2,-3],
             [-5,-3, 0,-1,-1,-1,-2, 0,-1, 1, 1,-1, 2, 5,-1,-3,-2,-2,-3,-2,-3],
             [-5,-1,-1,-1,-2,-1,-3,-2,-3,-2, 0,-2,-1,-1, 5, 1, 2, 1, 0,-1,-1],
             [-5,-1,-2,-1,-3,-1,-4,-3,-3,-3,-3,-3,-3,-3, 1, 4, 2, 3, 0,-1,-3],
             [-5,-1,-2,-1,-3,-1,-4,-3,-4,-3,-2,-3,-2,-2, 2, 2, 4, 1, 0,-1,-2],
             [-5,-1,-2, 0,-2, 0,-3,-3,-3,-2,-2,-3,-3,-2, 1, 3, 1, 4,-1,-1,-3],
             [-5,-2,-2,-2,-4,-2,-3,-3,-3,-3,-3,-1,-3,-3, 0, 0, 0,-1, 6, 3, 1],
             [-5,-2,-2,-2,-3,-2,-3,-2,-3,-2,-1, 2,-2,-2,-1,-1,-1,-1, 3, 7, 2],
             [-5,-2,-3,-2,-4,-3,-2,-4,-4,-3,-2,-2,-3,-3,-1,-3,-2,-3, 1, 2, 11]])
seq= ['X','C','S','T','P','A','G','N','D','E','Q','H','R','K','M','I','L','V','F','Y','W'] '''
#### Sara scoring matrix
s = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    	      [0, 9, 0, 0, 0, 1, 3, 0, 1, 0, 1, 1, 0, 1, 0, 0, 3, 2, 3, 0, 0],
              [0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 3, 0, 0, 0],
              [0, 0, 0, 9, 7, 0, 0, 0, 0, 1, 0, 0, 5, 0, 1, 1, 2, 3, 0, 0, 0],
              [0, 0, 0, 7, 9, 0, 0, 0, 0, 1, 0, 0, 1, 0, 5, 1, 1, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 9, 0, 3, 5, 0, 5, 3, 0, 3, 0, 0, 0, 0, 3, 3, 3],
              [0, 3, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 3, 0, 9, 0, 0, 0, 0, 0, 0, 3, 6, 0, 0, 0, 3, 3],
              [0, 1, 0, 0, 0, 5, 0, 0, 9, 0, 7, 3, 0, 3, 0, 0, 0, 0, 5, 2, 2],
              [0, 0, 0, 1, 1, 0, 0, 0, 0, 9, 0, 0, 1, 0, 3, 6, 0, 0, 0, 0, 1],
              [0, 1, 0, 0, 0, 5, 0, 0, 7, 0, 9, 3, 0, 3, 0, 0, 0, 0, 5, 2, 2],
              [0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 3, 9, 0, 1, 1, 0, 0, 0, 2, 2, 2],
              [0, 0, 0, 5, 1, 0, 0, 0, 0, 1, 0, 0, 9, 0, 5, 0, 2, 2, 0, 0, 0],
              [0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 3, 1, 0, 9, 0, 0, 0, 0, 2, 0, 0],
              [0, 0, 0, 1, 5, 0, 0, 3, 0, 3, 0, 1, 0, 0, 9, 2, 0, 0, 0, 0, 1],
              [0, 0, 0, 1, 1, 0, 0, 6, 0, 6, 0, 0, 0, 0, 2, 9, 0, 0, 0, 0, 1],
              [0, 3, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 9, 5, 0, 0, 3],
              [0, 2, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 5, 9, 0, 0, 5],
              [0, 3, 0, 0, 0, 3, 0, 0, 5, 0, 5, 2, 0, 2, 0, 0, 0, 0, 9, 1, 1],
              [0, 0, 0, 0, 0, 3, 0, 3, 2, 0, 2, 2, 2, 0, 0, 0, 0, 0, 1, 9, 3],
              [0, 0, 0, 0, 0, 3, 0, 3, 2, 1, 2, 2, 0, 0, 1, 1, 3, 5, 1, 3, 9]])
seq=['X','C','S','T','P','A','G','N','D','E','Q','H','R','K','M','I','L','V','F','Y','W']

f = open("/home/ajay/Desktop/Covarray_NGS_2/similarity_align/HCoV-NP_TomH-CD3-seq2num","r")
data = f.readlines()
f.close()

f = open("/home/ajay/Desktop/Covarray_NGS_2/similarity_align/HCoV-NP_TomH-CD3_align30.txt", "w+")
f.write("Sarascore_Cutoff score 30 or better\n")

for line in data:
    #a[0], a[1], a[2], a[3], a[4], a[5], a[6] = map(int, line.strip().split(',')) # 7-mmer
    a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9],a[10],a[11],a[12],a[13],a[14],a[15],a[16],a[17]= map(int, line.strip().split(',')) # 5-mmer
    #a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9] = map(int, line.strip().split(','))
    f.write("\n")
    inser = ("testsequence= "+
             seq[a[0]]+seq[a[1]]+seq[a[2]]+seq[a[3]]+seq[a[4]]+seq[a[5]]+seq[a[6]]+seq[a[7]]+seq[a[8]]+seq[a[9]]+
             seq[a[10]]+seq[a[11]]+seq[a[12]]+seq[a[13]]+seq[a[14]]+seq[a[15]]+seq[a[16]]+seq[a[17]]+'\n')
    #inser = ("testsequence= "+seq[a[0]]+seq[a[1]]+seq[a[2]]+seq[a[3]]+seq[a[4]]+seq[a[5]]+seq[a[6]]+seq[a[7]]+seq[a[8]]+seq[a[9]]+'\n')
    f.write(inser)
    
    
    for line2 in data:
        #b[0], b[1], b[2], b[3], b[4], b[5], b[6] = map(int, line2.strip().split(','))  # 7-mmer
        b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15],b[16],b[17] = map(int, line2.strip().split(','))  # 5-mmer
        #b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9] = map(int, line2.strip().split(','))
        score = sum(s[a[i], b[i]] for i in range(18)) #range(7)
        if score >= 30 and line2 != line:
            f.write("alignment=    ")
            f.write("".join(seq[b[i]] for i in range(18))) #range(7)
            f.write(" score=")
            f.write(str(score))
            f.write("\n")
'''
 # rooling tiles ####
for line in data:
    a[0], a[1], a[2], a[3], a[4], a[5], a[6] = map(int, line.strip().split(','))
    f.write("\n")
    inser = ("testsequence= "+seq[a[0]]+seq[a[1]]+seq[a[2]]+seq[a[3]]+seq[a[4]]+seq[a[5]]+seq[a[6]]+'\n')
    f.write(inser)
    for shift in range(-4, 5):
        shifted_b = np.roll(b, shift)
        score = sum(s[a[i], shifted_b[i]] for i in range(7))
        if score >= 20:
            f.write("alignment=    ")
            f.write("".join(seq[shifted_b[i]] for i in range(7)))
            f.write(" score=")
            f.write(str(score))
            f.write(" shift=")
            f.write(str(shift))
            f.write("\n")

'''
f.close()


'''

import itertools
import numpy as np

#seq_mapping = {'GGGGGGG': 0, 'SYQDNPY': 1, 'VVNPYAY': 2, 'AVVDEVH': 3, 'MLIKAIA': 4, 'LDRVAIF': 5, 'DVIHDNT': 6}


with open("list_of_sequences.txt", "r") as f:
    sequences = f.readlines()

with open("3CLPro_7mer_align.txt", "w+") as f:
    f.write("Sarascore_Cutoff score 20 or better\n")
    for seq_a, seq_b in itertools.combinations(sequences, 2):
        seq_a = seq_a.strip()
        seq_b = seq_b.strip()
        a = np.array([seq_mapping[seq_a[i:i+7]] for i in range(len(seq_a)-6)])
        b = np.array([seq_mapping[seq_b[i:i+7]] for i in range(len(seq_b)-6)])
        for i in range(len(a)):
            for j in range(len(b)):
                score = sum(s[a[i:i+7], b[j:j+7]])
                if score >= 20:
                    f.write("alignment=")
                    f.write(seq_b[j:j+7])
                    f.write(" score=")
                    f.write(str(score))
                    f.write("\n")

'''
