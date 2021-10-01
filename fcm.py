import numpy as np
import cv2
import sys
import random
import math
from matplotlib import pyplot as plt
from scipy import ndimage
from itertools import permutations

def distanciaEuclediana( (x,y,z),(x1,y1,z1)):
    return math.sqrt( math.pow((x-x1),2) + math.pow((y-y1),2) + math.pow((z-z1),2) )   

e1 = cv2.getTickCount()

img = cv2.imread('t1.jpg',1)
img1 = cv2.imread('t1.jpg',1)

print(img)

cv2.imshow('image',img)
cv2.waitKey(0) # se meter 0 fica para sempre
cv2.destroyAllWindows()

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
    v.append( (random.randrange(0,255), random.randrange(0,255),random.randrange(0,255) ) )
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
    print v
    flag=0
    # agrupar os pontos nos clusters
    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,c):
                res = 0.0
                contadorExcecao = 0
                aux1 = distanciaEuclediana(v[k],(img[i][j][0],img[i][j][1],img[i][j][2]) )
                if(aux1==0):            
                    vCont[k] += 1
                    continue  
    
                for m in range(0,c):
                    aux = distanciaEuclediana(v[m],(img[i][j][0],img[i][j][1],img[i][j][2]) )
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
        x,y,z,total = 0.0,0.0,0.0,0.0
        for i in range(0,height):    
            for j in range(0,width):
                x += img[i][j][0]*math.pow(classeCluster[i][j][k],q)
                y += img[i][j][1]*math.pow(classeCluster[i][j][k],q)
                z += img[i][j][2]*math.pow(classeCluster[i][j][k],q)
                total += math.pow(classeCluster[i][j][k],q)
                    
        if(total ==0):
            auxiliarx = 0
            auxiliary = 0
            auxiliarz = 0
        if(total != 0):
            auxiliarx = x / total
            auxiliary = y / total
            auxiliarz = z / total
        (valorAntigox,valorAntigoy,valorAntigoz) = v[k]
        if( abs( valorAntigox - auxiliarx) + abs(valorAntigoy - auxiliary) + abs(valorAntigoz - auxiliarz)  > epsilon   ) :
            v[k] =  (auxiliarx,auxiliary,auxiliarz)
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
        (v1,v2,v3) = v[i]
        (v11,v22,v33) = v[k]
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
            (valorClusterx,valorClustery,valorClusterz) = v[k]
            contaAuxiliar = abs(valorPontox-valorClusterx) + abs(valorPontoy-valorClustery) + abs(valorPontoz-valorClusterz)
            eqBaixo = eqBaixo + pow( classeCluster[i][j][k],q) * pow(contaAuxiliar,2)
        
xb = eqCima / eqBaixo
print "O valor de Xie-Beni index:"
print xb

'''

#-------------------------------------------------#  imagem a mao
listaCores = []
for i in range(0,c):
    listaCores.append( [] )

contadorCluster = 0
for i in range(0,height):
    print listaCores
    for j in range(0,width):
        (a,b,c) = img1[i][j]
        x = [a,b,c]
        if( not(x in listaCores) ):
            #   listaCores[contadorCluster].append(x)
            listaCores[contadorCluster].append( a )
            listaCores[contadorCluster].append( b )
            listaCores[contadorCluster].append( c )
            contadorCluster += 1
            #   print x
   
listaHumana = []
for i in range(0,height):
    listaHumana.append( [] )
    for j in range(0,width):
        (a,b,c) = img1[i][j]
        x = [a,b,c]
        if( x == listaCores[0]):
            listaHumana[i].append( 0 )
        if( x == listaCores[1]):
            listaHumana[i].append( 1 )     
        if( x == listaCores[2]):
            listaHumana[i].append( 2 )
        if( x == listaCores[3]):
            listaHumana[i].append( 3 )
        if( x == listaCores[4]):
            listaHumana[i].append( 4 )
        if( x == listaCores[5]):
            listaHumana[i].append( 5 )
        if( x == listaCores[6]):
            listaHumana[i].append( 6 )


#print listaHumana
#-------------------------------------------------#  segmentada

height, width = img.shape[:2] # posso acrescentar outra var e tirar [:2].(ch)
listaPrograma = []
for i in range(0,height):
    listaPrograma.append( [] )
    for j in range(0,width):
        if( classeCluster[i][j][0] == max(classeCluster[i][j]) ): 
            listaPrograma[i].append( 0 )
        if( classeCluster[i][j][1] == max(classeCluster[i][j]) ):    
            listaPrograma[i].append( 1 )

        if( classeCluster[i][j][2] == max(classeCluster[i][j]) ):  
            listaPrograma[i].append( 2 )
        if( classeCluster[i][j][3] == max(classeCluster[i][j]) ):  
            listaPrograma[i].append( 3 )
        if( classeCluster[i][j][4] == max(classeCluster[i][j]) ):
            listaPrograma[i].append( 4 )
        if( classeCluster[i][j][5] == max(classeCluster[i][j]) ):    
            listaPrograma[i].append( 5 )
        if( classeCluster[i][j][6] == max(classeCluster[i][j]) ):    
            listaPrograma[i].append( 6 )   

#print listaPrograma
#----------------------------------------------------# mapear
height, width = img1.shape[:2] # posso acrescentar outra var e tirar [:2].(ch)
perms = [''.join(p) for p in permutations('01234')]   

output = []
total = 0
for P in perms:
    print P
    correct = 0
    for i in range(0,height):
        for j in range(0,width):
           #if( listaHumana[i][j] == int( P[ listaHumana[i][j] ] ) ):
           #     correct += 1
           print v
           print listaPrograma[i][j]
           if( listaPrograma[i][j] == int( p[int(listaPrograma[i][j]) ] ) ):
                 correct += 1
    output.append(correct)


print max(output)
print n

sa =  ( float( max(output) ) / n ) * 100
print "a precisao da segmentacao e"
print sa
'''

listaHumana = []
for i in range(0,height):
    listaHumana.append( [] )
    for j in range(0,width):
        (a,b,c) = img1[i][j]
        x = [a,b,c]
        if( x == v[0]):
            listaHumana[i].append( 0 )
        if( x == v[1]):
            listaHumana[i].append( 1 )        
        if( x == v[2]):
            listaHumana[i].append( 2 )

print listaHumana


print v
menor = 0
menorc = 0
maior = 0
maiorc = 0
for i in v:
    (a,b,d) = i
    if( menor > a + b + d):
        menor = a + b + d
        menorc = c
    if( maior < a + b + d):
        maior = a + b + d
        maiorc = c    

medioc = 0
for i in range(0,3):
    if( i != maiorc and i != menorc):
        medioc = i
v1 = []
#v1.append(v[menorc])
#v1.append(v[medioc])
v1.append(menorc)
v1.append(medioc)
print v1

zonaQuente = 0
zonaNormal = 0
for i in range(0,height):
    for j in range(0,width):
        if( max(classeCluster[i][j]) == v1[0]):
            zonaQuente += 1
        if( max(classeCluster[i][j]) == v1[1]):
            zonaNormal += 1

print "A estimativa de leite para a proporcao de mama e de"
if( zonaNormal == 0):
    zonaNormal = 1
print float( zonaQuente) / zonaNormal

'''
2.032919 , 7.86745, 68.589        azul    
254.3232, 189.64671 15.665         medio
252.7957, 240.610,113.1974        quente      


menor preto
'''

                        #pintar para testar
teste = np.zeros((width,height,3), np.uint8)
for i in range(0,height):
    for j in range(0,width):
        if( classeCluster[i][j][0] == max(classeCluster[i][j]) ):  # se mais proximo 1 cluster -> azul
            cv2.line(teste,(i,j),(i,j),v[0],1)

        if( classeCluster[i][j][1] == max(classeCluster[i][j]) ):    
            cv2.line(teste,(i,j),(i,j),v[1],1)

        if( c >= 3):
            if( classeCluster[i][j][2] == max(classeCluster[i][j]) ):  
                cv2.line(teste,(i,j),(i,j),v[2],1)

plt.figure()
imAx = plt.imshow(teste)
lena = ndimage.rotate(teste, 90)
plt.imshow(lena, interpolation='nearest', origin='lower')
plt.axis('off')
plt.show()

cv2.imwrite('imagem.png',lena)
