import numpy as np
import cv2
import sys
import random
import math
from matplotlib import pyplot as plt
from scipy import ndimage

e1 = cv2.getTickCount()

img1 = cv2.imread('g.png',1)
img = cv2.imread('h.png',1)


c = 7

height, width = img.shape[:2] # posso acrescentar outra var e tirar [:2].(ch)
n = height * width

#-------------------------------------------------#  imagem a mao
listaCores = []
for i in range(0,c):
    listaCores.append( [] )

contadorCluster = 0
for i in range(0,height):
    print listaCores
    for j in range(0,width):
        (a,b,c) = img[i][j]
        x = [a,b,c]
        if( not(x in listaCores) ):
            #   listaCores[contadorCluster].append(x)
            listaCores[contadorCluster].append( a )
            listaCores[contadorCluster].append( b )
            listaCores[contadorCluster].append( c )
            contadorCluster += 1
            print x
   
listaHumana = []
for i in range(0,height):
    listaHumana.append( [] )
    for j in range(0,width):
        (a,b,c) = img[i][j]
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
print listaHumana
#-------------------------------------------------#  segmentada

height, width = img1.shape[:2] # posso acrescentar outra var e tirar [:2].(ch)
listaPrograma = []
for i in range(0,height):
    listaPrograma.append( [] )
    for j in range(0,width):
        if( classeCluster[i][j][0] == max(classeCluster[i][j]) ): 
            listaPrograma[i].append( 0 )
        if( classeCluster[i][j][1] == max(classeCluster[i][j]) ):    
            listaPrograma[i].append( 1 )
        if( c >= 3):
            if( classeCluster[i][j][2] == max(classeCluster[i][j]) ):  
                listaPrograma[i].append( 2 )
        if( c >= 4):
            if( classeCluster[i][j][3] == max(classeCluster[i][j]) ):  
                listaPrograma[i].append( 3 )
        if( c >= 5):
            if( classeCluster[i][j][4] == max(classeCluster[i][j]) ):
                listaPrograma[i].append( 4 )
        if( c >= 6):
            if( classeCluster[i][j][5] == max(classeCluster[i][j]) ):    
                listaPrograma[i].append( 5 )
        if( c >= 7):
            if( classeCluster[i][j][6] == max(classeCluster[i][j]) ):    
                listaPrograma[i].append( 6 )   
        if( c >= 8):
            if( classeCluster[i][j][7] == max(classeCluster[i][j]) ):  
                listaPrograma[i].append( 7 )
        if( c >= 9):
            if( classeCluster[i][j][8] == max(classeCluster[i][j]) ):
                listaPrograma[i].append( 8 )
        if( c >= 10):
            if( classeCluster[i][j][9] == max(classeCluster[i][j]) ):   
                listaPrograma[i].append( 9 )

#-----------------------------------------------------# mapear
melhorCaso = 0
for k in range(0,c):
    for l in range(0,c): 
        pixelsCertos = 0
        for i in range(0,height):
            for j in range(0,width):
                if( listaPrograma[i][j] == i and listaHumano[i][j] == h):
                    pixelsCertos += 1

        sa =  ( pixelsCertos / n) * 100
        if( melhorCaso < sa):
            melhorCaso = sa
        print "a precisao da segmentacao:"
        print sa

print "O melhor caso da precisao da segmentacao:"
print melhorCaso
