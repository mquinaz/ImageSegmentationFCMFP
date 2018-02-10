import numpy as np
import cv2
import sys
import random
import math
from matplotlib import pyplot as plt

def distanciaEuclediana( (x,y),(x1,y1)):
    return math.sqrt( math.pow((x-x1),2) + math.pow((y-y1),2))

e1 = cv2.getTickCount()
                # aplicar FCM

c = int(input('Insira c: '))
q = float(input('Insira q: '))

height, width = 3,3
n = height * width

#criar os clusters
v = [] 
for i in range(0,c):
    v.append( (random.randrange(0,height), random.randrange(0,width)) )
    print v[i]

centroCluster = []
classeCluster = []
for i in range(0,height):
    classeCluster.append( [] )
    for j in range(0,width):
        classeCluster[i].append( [] )
        for k in range(0,c):
            classeCluster[i][j].append( 0 )

#classeCluster[0][2][1] = 32
#print classeCluster[0]

potencia = 2 / (q-1)
flag = 1
while flag == 1:
    flag=0
    # agrupar os pontos nos clusters
    for i in range(0,height):
            for j in range(0,width):
                for k in range(0,c):
                    res = 0.0
                    for m in range(0,c):
                        aux = distanciaEuclediana(v[m],(i,j))
                        if(aux == 0):    
                            classeCluster[i][j][k] = 1
                            continue
                        res +=   math.pow( ( distanciaEuclediana(v[k],(i,j)) ) / aux  , potencia)
                    if(res==0):
                        classeCluster[i][j][k] = 0
                        continue
                    #res = round(res,2)
                    classeCluster[i][j][k] =  1.0 / res

    # calcular os centros dos clusters
    for k in range(0,c):
        x,y,total = 1,1,1
        for i in range(0,height):    
            for j in range(0,width):
                x += i*math.pow(classeCluster[i][j][k],q)
                y += j*math.pow(classeCluster[i][j][k],q)
                total += math.pow(classeCluster[i][j][k],q)
                    

        auxiliarx = round( (x-1) / (total-1) , 10)
        auxiliary = round( (y-1) / (total-1) , 10)
        if( v[k] != (auxiliarx,auxiliary) ) :
            v[k] =  (auxiliarx,auxiliary)
            flag = 1
    print v
    print classeCluster

e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print time

                        #pintar para testar
teste = np.zeros((height,width,3), np.uint8)
for i in range(0,height):
    for j in range(0,width):
        if(c==3):
            if( classeCluster[i][j][2] > classeCluster[i][j][0] and classeCluster[i][j][2] > classeCluster[i][j][1]):    
                cv2.line(teste,(i,j),(i,j),(0,255,0),1) 
            if( classeCluster[i][j][0] > classeCluster[i][j][1] and classeCluster[i][j][0] > classeCluster[i][j][2]):  
                cv2.line(teste,(i,j),(i,j),(255,0,0),1)
            if( classeCluster[i][j][1] > classeCluster[i][j][0] and classeCluster[i][j][1] > classeCluster[i][j][2]):    
                cv2.line(teste,(i,j),(i,j),(0,0,255),1)
            continue
        if( classeCluster[i][j][0] >= classeCluster[i][j][1]):  # se mais proximo 1 cluster -> azul
            cv2.line(teste,(i,j),(i,j),(255,0,0),1)
        if( classeCluster[i][j][1] > classeCluster[i][j][0]):    
            cv2.line(teste,(i,j),(i,j),(0,0,255),1)
res = cv2.resize(teste,None,fx=25, fy=25, interpolation = cv2.INTER_CUBIC)
cv2.imshow('image',res)
cv2.waitKey(0) # se meter 0 fica para sempre
cv2.destroyAllWindows()
