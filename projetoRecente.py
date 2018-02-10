import numpy as np
import cv2
import sys
import random
import math
from matplotlib import pyplot as plt
from skimage import io, color
from scipy import ndimage

def distanciaEuclediana( (x,y),(x1,y1)):
    return math.sqrt( math.pow((x-x1),2) + math.pow((y-y1),2) )   

e1 = cv2.getTickCount()

#img = cv2.imread('um.jpeg',1) 
img = cv2.imread('regioes.png',1) 
print img[0][0]

cv2.imshow('image',img)
cv2.waitKey(2500) # se meter 0 fica para sempre
cv2.destroyAllWindows()

b, g, r = cv2.split(img)
im = cv2.merge((r,g,b))

lab = color.rgb2lab(im)
print "MUDA DE MODELO COR1"
print lab[0][0]

                # aplicar FCM
c = int(input('Insira c: '))
q = float(input('Insira q: '))
epsilon = float(input('Insira epsilon: '))

height, width = img.shape[:2] # posso acrescentar outra var e tirar [:2].(ch)
n = height * width

#criar os clusters
v = [] 
vCont = []
for i in range(0,c):
    v.append( (random.randrange(-127,127), random.randrange(-127,127) ) )
    vCont.append(0)
    print v[i]

centroCluster = []
classeCluster = []
for i in range(0,height):
    classeCluster.append( [] )
    for j in range(0,width):
        classeCluster[i].append( [] )
        for k in range(0,c):
            classeCluster[i][j].append( 0 )

potencia = 2 / (q-1)
flag = 1
while flag == 1:
    flag=0
    # agrupar os pontos nos clusters
    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,c):
                res = 0.0
                contadorExcecao = 0
                aux1 = distanciaEuclediana(v[k],(lab[i][j][1],lab[i][j][2]) )
                if(aux1==0):            
                    vCont[k] += 1
                    continue  
    
                for m in range(0,c):
                    aux = distanciaEuclediana(v[m],(lab[i][j][1],lab[i][j][2]) )
                    if(aux == 0):    
                        continue
                    res += math.pow( aux1/aux  , potencia)

                if(res==0):
                    classeCluster[i][j][k] = -1
                    continue
                classeCluster[i][j][k] =  1.0 / res
    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,c):
                if(classeCluster[i][j][k] == -1):
                    classeCluster[i][j][k] = 1 / vCont[k]
                    for k in range(0,c):
                        if(classeCluster[i][j][k] == -1):
                            classeCluster[i][j][k] = 0
    print v
    # calcular os centros dos clusters
    for k in range(0,c):
        x,y,total = 1,1,1
        for i in range(0,height):    
            for j in range(0,width):
                x += lab[i][j][1]*math.pow(classeCluster[i][j][k],q)
                y += lab[i][j][2]*math.pow(classeCluster[i][j][k],q)
                total += math.pow(classeCluster[i][j][k],q)            

        auxiliarx = (x-1) / (total-1)
        auxiliary = (y-1) / (total-1)
        (valorAntigox,valorAntigoy) = v[k]
        if( abs( valorAntigox - auxiliarx) + abs(valorAntigoy - auxiliary) > epsilon   ):
            v[k] =  (auxiliarx,auxiliary)
            flag = 1


e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print "o tempo de execucao:"
print time


                        # Xie-Beni index
minimo= 99999999999999
for i in range(0,c):
    for k in range(0,c):
        if( i == k):
            continue
        (v1,v2) = v[i]
        (v11,v22) = v[k]
        minimoContaAux = abs(v1-v11) + abs(v2-v22) 
        minimoAux = pow( minimoContaAux , 2)
        if( minimo > minimoAux ):
             minimo = minimoAux
    
eqCima = n * minimo
eqBaixo = 0
for k in range(0,c):
    for i in range(0,height):    
        for j in range(0,width):
            (valorPontox,valorPontoy) = lab[i][j][1],lab[i][j][2]
            (valorClusterx,valorClustery) = v[k]
            contaAuxiliar = abs(valorPontox-valorClusterx) + abs(valorPontoy-valorClustery)
            eqBaixo = eqBaixo + pow( classeCluster[i][j][k],q) * pow(contaAuxiliar,2)
        
xb = eqCima / eqBaixo
print "O valor de Xie-Beni index:"
print xb

lab = color.lab2rgb(im)
                        #pintar para testar
teste = np.zeros((width,height,3), np.uint8)
for i in range(0,height):
    for j in range(0,width):
        if( classeCluster[i][j][0] == max(classeCluster[i][j]) ):  # se mais proximo 1 cluster -> azul
            cv2.line(teste,(i,j),(i,j),(255,0,0),1)
        if( classeCluster[i][j][1] == max(classeCluster[i][j]) ):    
            cv2.line(teste,(i,j),(i,j),(0,0,255),1)#red
        if( c >= 3):
            if( classeCluster[i][j][2] == max(classeCluster[i][j]) ):    
                cv2.line(teste,(i,j),(i,j),(0,255,0),1) #green
        if( c >= 4):
            if( classeCluster[i][j][3] == max(classeCluster[i][j]) ):    
                cv2.line(teste,(i,j),(i,j),(255,255,0),1) #yellow
        if( c >= 5):
            if( classeCluster[i][j][4] == max(classeCluster[i][j]) ):    
                cv2.line(teste,(i,j),(i,j),(128,128,128),1) #grey
        if( c >= 6):
            if( classeCluster[i][j][5] == max(classeCluster[i][j]) ):    
                cv2.line(teste,(i,j),(i,j),(127,0,255),1) #pink
        if( c >= 7):
            if( classeCluster[i][j][6] == max(classeCluster[i][j]) ):    
                cv2.line(teste,(i,j),(i,j),(255,0,127),1) #purple
        if( c >= 8):
            if( classeCluster[i][j][7] == max(classeCluster[i][j]) ):    
                cv2.line(teste,(i,j),(i,j),(255,255,0),1) # light blue
        if( c >= 9):
            if( classeCluster[i][j][8] == max(classeCluster[i][j]) ):    
                cv2.line(teste,(i,j),(i,j),(0,128,255),1)  #orange
        if( c >= 10):
            if( classeCluster[i][j][9] == max(classeCluster[i][j]) ):    
                cv2.line(teste,(i,j),(i,j),(0,0,0),1) #black

plt.figure()
imAx = plt.imshow(teste)
lena = ndimage.rotate(teste, 90)
plt.imshow(lena, interpolation='nearest', origin='lower')
plt.axis('off')
plt.show()

cv2.imwrite('imagemLab.png',lena)

