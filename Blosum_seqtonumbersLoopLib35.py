# -*- coding: utf-8 -*-
bv=[]
nr=0
nr1=1
fr1=0.12
myline = []
myline2 = []
nrline = []
a = "22"
test = "O"
delimit = ','
seq_file = '/home/ajay/Desktop/Covarray_NGS_2/similarity_align/HCoV-NP_TomH-CD3'
num_file = '/home/ajay/Desktop/Covarray_NGS_2/similarity_align/HCoV-NP_TomH-CD3-seq2num'
with open(seq_file,'r') as infile, open(num_file,'w+') as out1:
    for line in infile:
        num=''
        myline = line.rstrip()
        b=myline
        #print("b=",myline)
        if b != '':
            x=0
            for x in range(len(b)):
                #print(len(b))
                
                test = b[x]
                

                if test == "X":
                    a = "0"
                if test == "C":
                    a = "1"
                if test == "S":
                    a = "2"
                if test == "T":
                    a = "3"
                if test == "P":
                    a = "4"
                if test == "A":
                    a = "5"
                if test == "G":
                    a = "6"
                if test == "N":
                    a = "7"
                if test == "D":
                    a = "8"
                if test == "E":
                    a = "9"
                if test == "Q":
                    a = "10"
                if test == "H":
                    a = "11"
                if test == "R":
                    a = "12"
                if test == "K":
                    a = "13"
                if test == "M":
                    a = "14"
                if test == "I":
                    a = "15"
                if test == "L":
                    a = "16"
                if test == "V":
                    a = "17"
                if test == "F":
                    a = "18"
                if test == "Y":
                    a = "19"
                if test == "W":
                    a = "20"
                if x!=len(b):
                    num += a
                    num += delimit                    
                else:
                    print(x)
                    num += a
                    
                x =x + 1
        in_text = num[:-1]+'\n'
        out1.write(in_text)
