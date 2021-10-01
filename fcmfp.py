import numpy as np
import cv2
import sys
import random
import math
from matplotlib import pyplot as plt
from scipy import ndimage

def distanciaEuclediana( (x,y,z,r),(x1,y1,z1,r1)):
    return math.sqrt( math.pow((x-x1),2) + math.pow((y-y1),2) + math.pow((z-z1),2) + math.pow((r-r1),2) )

e1 = cv2.getTickCount()

img = cv2.imread('um.jpeg',1)
print(img)

                # aplicar FCM
c = int(input('Insira c: '))
q = float(input('Insira q: '))
epsilon = float(input('Insira epsilon: '))
zeta = float(input('Insira zeta : '))
potencia = 2 / (q-1)

height, width = img.shape[:2] # posso acrescentar outra var e tirar [:2].(ch)
n = height * width

                    # baricentro do dataset 
valorx = 0
valory = 0
valorz = 0
valorR = 1000             # valor R+1 definido
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

#criar os clusters
v = [] 
vCont = []
for i in range(0,c):
    v.append( (random.randrange(0,255), random.randrange(0,255),random.randrange(0,255),0 ) )
    vCont.append(0)
    print v[i]

while c > 2:
    centroCluster = []
    classeCluster = []
    for i in range(0,height):
        classeCluster.append( [] )
        for j in range(0,width):
            classeCluster[i].append( [] )
            for k in range(0,c):
                classeCluster[i][j].append( 0 )


    flag = 1
    while flag == 1:
        print v
        flag=0
        # agrupar os pontos nos clusters
        for i in range(0,height):
            for j in range(0,width):
                for k in range(0,c):
                    res = 0.0
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
                            #for k in range(0,c):
                            #if(classeCluster[i][j][k] == -1):
                                #classeCluster[i][j][k] = 0    esta bem? 

        # calcular os centros dos clusters
        for k in range(0,c):
            x,y,z,total = 0.0,0.0,0.0,1.0
            for i in range(0,height):    
                for j in range(0,width):
                    x += img[i][j][0]*math.pow(classeCluster[i][j][k],q)
                    y += img[i][j][1]*math.pow(classeCluster[i][j][k],q)
                    z += img[i][j][2]*math.pow(classeCluster[i][j][k],q)
                    #r += img[i][j][3]*math.pow(classeCluster[i][j][k],q)     img[i][j][3] = 0  (=) r = 0
                    total += math.pow(classeCluster[i][j][k],q) + zeta

            auxiliarx = ( x + valorx * zeta) / (total-1)
            auxiliary = ( y + valory * zeta) / (total-1)
            auxiliarz = ( z + valorz * zeta) / (total-1) 
            auxiliarR = ( valorR * zeta) / (total-1)  # r = 0
            (valorAntigox,valorAntigoy,valorAntigoz,valorAntigoR) = v[k]
            if( abs( valorAntigox - auxiliarx) + abs(valorAntigoy - auxiliary) + abs(valorAntigoz - auxiliarz) + abs(valorAntigoR - auxiliarR)  > epsilon   ) :
                v[k] =  (auxiliarx,auxiliary,auxiliarz,auxiliarR)
                flag = 1 
                           

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

                  # exluir clusters  -   2 abordagem
 
    vetorAux = []
    for k in range(0,c):
        vetorAux.append(0)
    indiceAux = 0
    for i in range(0,height):
        maxAuxCluster = 0
        for j in range(0,width):
            for k in range(0,c):
                if( maxAuxCluster < classeCluster[i][j][k]):            # <= ou <
                    maxAuxCluster = classeCluster[i][j][k]    

                    indiceAuxCluster = k
            if( vetorAux[indiceAuxCluster] == 0):
                vetorAux[indiceAuxCluster] = 1
            
    
    zeta += 0.01                            # incrementar zeta
    for k in range(c-1,0,-1): #c-1 porque se 5 clusters = v[4]
        if( vetorAux[k] == 0):
            #del v[k]             
            v.pop(k)
            c -= 1

    e2 = cv2.getTickCount()
    time = (e2 - e1)/ cv2.getTickFrequency()
    print "o tempo de execucao:"
    print time


#pintar para testar
teste = np.zeros((width,height,3), np.uint8)
for i in range(0,height):
    for j in range(0,width):               
        if( classeCluster[i][j][0] == max(classeCluster[i][j]) ):  # se mais proximo 1 cluster -> v[0]
            (a,b,c,lixo) = v[0]
            cv2.line(teste,(i,j),(i,j),(a,b,c),1)   
        if( classeCluster[i][j][1] == max(classeCluster[i][j]) ):  # se mais proximo 1 cluster -> v[0]
            (a,b,c,lixo) = v[1]
            cv2.line(teste,(i,j),(i,j),(a,b,c),1)   


plt.figure()
imAx = plt.imshow(teste)
lena = ndimage.rotate(teste, 90)
plt.imshow(lena, interpolation='nearest', origin='lower')
plt.axis('off')
plt.show()

cv2.imwrite('imagem.png',lena)

