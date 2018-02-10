import numpy as np
import cv2
import sys
import random
import math
from matplotlib import pyplot as plt
from scipy import ndimage

def distanciaEuclediana( (x,y,z,r),(x1,y1,z1,r1)):
    return math.sqrt( math.pow((x-x1),2) + math.pow((y-y1),2) + math.pow((z-z1),2) + math.pow((r-r1),2) )   # contabilizar r?

e1 = cv2.getTickCount()

img = cv2.imread('um.jpeg',1)
#img = cv2.imread('regioes.png',1)
print(img)

cv2.imshow('image',img)
cv2.waitKey(0) # se meter 0 fica para sempre
cv2.destroyAllWindows()

                # aplicar FCM
c = int(input('Insira c: '))
q = float(input('Insira q: '))
epsilon = float(input('Insira epsilon: '))
eta = float(input('Insira zeta : '))
potencia = 2 / (q-1)

height, width = img.shape[:2] # posso acrescentar outra var e tirar [:2].(ch)
n = height * width

#criar os clusters
v = [] 
vCont = []
for i in range(0,c):
    v.append( (random.randrange(0,255), random.randrange(0,255),random.randrange(0,255),0 ) )
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


                    # baricentro do dataset 
valorx = 0
valory = 0
valorz = 0
valorR = 10             # valor R+1 definido
contadorBaricentro = 0
for i in range(0,height):
    for j in range(0,width):
        valorx += img[i][j][0]
        valory += img[i][j][1]
        valorz += img[i][j][2]
        contadorBaricentro += 1

valorx = valorx / contadorBaricentro
valory = valory / contadorBaricentro
valorz = valorz / contadorBaricentro

flag = 1
while flag == 1:
    print v
    flag=0
    # agrupar os pontos nos clusters
    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,c):
                res = 0.0
                contadorExcecao = 0
                aux1 = distanciaEuclediana(v[k],(img[i][j][0],img[i][j][1],img[i][j][2],0) )
                if(aux1==0):            
                    vCont[k] += 1
                    continue  
    
                for m in range(0,c):
                    aux = distanciaEuclediana(v[m],(img[i][j][0],img[i][j][1],img[i][j][2],0) )
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

    # calcular os centros dos clusters
    for k in range(0,c):
        x,y,z,total = 0.0,0.0,0.0,1.0
        for i in range(0,height):    
            for j in range(0,width):
                x += img[i][j][0]*math.pow(classeCluster[i][j][k],q)
                y += img[i][j][1]*math.pow(classeCluster[i][j][k],q)
                z += img[i][j][2]*math.pow(classeCluster[i][j][k],q)
                #r += img[i][j][3]*math.pow(classeCluster[i][j][k],q)     img[i][j][3] = 0  (=) r = 0
                total += math.pow(classeCluster[i][j][k],q) + eta

        auxiliarx = ( x + valorx * eta) / (total-1)
        auxiliary = ( y + valory * eta) / (total-1)
        auxiliarz = ( z + valorz * eta) / (total-1) 
        auxiliarR = ( valorR * eta) / (total-1)  # r = 0
        (valorAntigox,valorAntigoy,valorAntigoz,valorAntigoR) = v[k]
        if( abs( valorAntigox - auxiliarx) + abs(valorAntigoy - auxiliary) + abs(valorAntigoz - auxiliarz) + abs(valorAntigoR - auxiliarR)  > epsilon   ) :
            v[k] =  (auxiliarx,auxiliary,auxiliarz,auxiliarR)
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
        (v1,v2,v3,variavelLixo) = v[i]     #ignorar a coordenada R+1
        (v11,v22,v33,variavelLixo) = v[k]
        minimoContaAux = abs(v1-v11) + abs(v2-v22) + abs(v3-v33)
        minimoAux = pow( minimoContaAux , 2)
        if( minimo > minimoAux ):
             minimo = minimoAux
    
eqCima = n * minimo
eqBaixo = 0
for k in range(0,c):
    for i in range(0,height):    
        for j in range(0,width):
            (valorPontox,valorPontoy,valorPontoz) = img[i][j][0],img[i][j][1],img[i][j][2]
            (valorClusterx,valorClustery,valorClusterz,variavelLixo) = v[k] #ignorar a coordenada R+1
            contaAuxiliar = abs(valorPontox-valorClusterx) + abs(valorPontoy-valorClustery) + abs(valorPontoz-valorClusterz)
            eqBaixo = eqBaixo + pow( classeCluster[i][j][k],q) * pow(contaAuxiliar,2)
        
xb = eqCima / eqBaixo
print "O valor de Xie-Beni index:"
print xb
                        #pintar para testar
teste = np.zeros((width,height,3), np.uint8)
for i in range(0,height):
    for j in range(0,width):                # os 2 primeiros nao tem if porque no min  2 clusters
        if( classeCluster[i][j][0] == max(classeCluster[i][j]) ):  # se mais proximo 1 cluster -> azul
            (a,b,c,lixo) = v[0]
            cv2.line(teste,(i,j),(i,j),(a,b,c),1)
        if( classeCluster[i][j][1] == max(classeCluster[i][j]) ):    
            (a,b,c,lixo) = v[1]
            cv2.line(teste,(i,j),(i,j),(a,b,c),1)#red
        if( c >= 3):
            if( classeCluster[i][j][2] == max(classeCluster[i][j]) ):   
                (a,b,c,lixo) = v[2] 
                cv2.line(teste,(i,j),(i,j),(a,b,c),1) #green
        if( c >= 4):
            if( classeCluster[i][j][3] == max(classeCluster[i][j]) ):   
                (a,b,c,lixo) = v[3]  
                cv2.line(teste,(i,j),(i,j),(a,b,c),1) #yellow
        if( c >= 5):
            if( classeCluster[i][j][4] == max(classeCluster[i][j]) ):    
                (a,b,c,lixo) = v[4] 
                cv2.line(teste,(i,j),(i,j),(a,b,c),1) #grey
        if( c >= 6):
            if( classeCluster[i][j][5] == max(classeCluster[i][j]) ):    
                (a,b,c,lixo) = v[5] 
                cv2.line(teste,(i,j),(i,j),(a,b,c),1) #pink
        if( c >= 7):
            if( classeCluster[i][j][6] == max(classeCluster[i][j]) ):    
                (a,b,c,lixo) = v[6] 
                cv2.line(teste,(i,j),(i,j),(a,b,c),1) #purple
        if( c >= 8):
            if( classeCluster[i][j][7] == max(classeCluster[i][j]) ):  
                (a,b,c,lixo) = v[7]   
                cv2.line(teste,(i,j),(i,j),(a,b,c),1) # light blue
        if( c >= 9):
            if( classeCluster[i][j][8] == max(classeCluster[i][j]) ):    
                (a,b,c,lixo) = v[8] 
                cv2.line(teste,(i,j),(i,j),(a,b,c),1)  #orange
        if( c >= 10):
            if( classeCluster[i][j][9] == max(classeCluster[i][j]) ):    
                (a,b,c,lixo) = v[9] 
                cv2.line(teste,(i,j),(i,j),(a,b,c),1) #black

#cv2.imshow('image',teste)
#cv2.waitKey(0) # se meter 0 fica para sempre
#cv2.destroyAllWindows() 

plt.figure()
imAx = plt.imshow(teste)
lena = ndimage.rotate(teste, 90)
plt.imshow(lena, interpolation='nearest', origin='lower')
plt.axis('off')
plt.show()

cv2.imwrite('imagem.png',lena)

