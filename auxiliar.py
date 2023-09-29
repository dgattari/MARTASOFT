#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:28:03 2022

@author: dgattari
"""
    
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import feature
img = cv2.imread('2_Cx43_cut3')
cut1 = img[7100:7600,3400:4500,:]
cut2 = img[5500:6000,1000:2100,:]
cut3 = img[4000:4500,2200:3300,:]
plt.imshow(cut3)

cv2.imwrite('2_Cx43_cut3.tif', cut3)


img_mono = cv2.imread('2_Cx43_cut3',0) # lee la imagen monocroma
img_mono_equ = cv2.equalizeHist(img_mono) # la ecualiza
plt.hist(img_mono.ravel(), 256,[0,256] )
plt.hist(img_mono_equ.ravel(), 256,[0,256] )


plt.figure(2)
plt.subplot(1,2,1),plt.imshow(img_mono, cmap='gray'),plt.title("sin ecualizar")
plt.subplot(1,2,2),plt.imshow(img_mono_equ, cmap='gray'),plt.title("ecualizada")
plt.show()

tresh_c1 = 200
umbralizada = cv2.threshold(img_mono_equ,tresh_c1,255,cv2.THRESH_BINARY)[1]
kernel = np.ones((5,5),np.uint8)
umbralizada=cv2.morphologyEx(umbralizada, cv2.MORPH_CLOSE, kernel,iterations=2) # realiza apertura en imagen binarizada (para e esta accion está anulada ya que el kernel es de 1x1)


plt.figure(3)
plt.imshow(umbralizada, cmap='gray')
plt.show()

ancho=99 #entero impar, pues se considera un área de ancho x ancho. defino entornos de 21x21
offset=20 #se le resta a la media del entorno para dar el umbral. Mientras más alto más blancos en la imagen

umbralizada2 = cv2.adaptiveThreshold(img_mono_equ,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,ancho,offset)


plt.figure(figsize=(12,10))
plt.subplot(2,1,1), plt.imshow(umbralizada,cmap='gray'),plt.title('Umbralizado simple'),plt.xticks([]), plt.yticks([])
plt.subplot(2,1,2),plt.imshow(umbralizada2,cmap='gray'),plt.title('Umbralizado adaptativo'),plt.xticks([]), plt.yticks([])
plt.show()

##############

up_mask_g=255
dwn_mask_g=250   
lower = np.array([0,dwn_mask_g,0])# [0,250,0]
upper = np.array([0,up_mask_g,0])# [0,255,0]

imgserca = cv2.imread("imgserca.tif",0) #fondo negro
imgcx = cv2.imread("imgcx.tif",0)
imgwga = cv2.imread("imgwga.tif",0)
img3comb = cv2.imread("blacksub.tif") #fondo negro
img3comb[:,:,2]=cv2.add(imgwga,imgcx) # cv2.add combina las imagenes
img3comb[:,:,1]=cv2.add(imgserca,imgcx)
img3comb[:,:,0]=imgcx


mask_green = cv2.inRange(img3comb, lower, upper)
contours2, hierarchy = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for j in range(0,len(contours2)-1,1):        
    epsilon = 0.1*cv2.arcLength(contours2[j],True)
    contours2[j] = cv2.approxPolyDP(contours2[j],epsilon,True)
    
epsilon = 0.1*cv2.arcLength(contours2[j],True)
contours2[3] = cv2.approxPolyDP(contours2[3],epsilon,True)

##########

horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
imgc2 = cv2.imread('2_Cx43_cut2_c2.tif',0)
tresh_c2 = 12
thresh1cx = cv2.threshold(imgc2,tresh_c2,255,cv2.THRESH_BINARY)[1]
kernel_noise_remove_c2=np.ones((2, 2),np.uint8)
ercx=cv2.morphologyEx(thresh1cx, cv2.MORPH_OPEN, kernel_noise_remove_c2)
gimgcx=cv2.dilate(ercx,horizontalStructure,iterations = 5)

#########

binary = cv2.imread('out_quantif_1670008057/2_Cx43_cut2_1670008057_cell_mask_nb.tif',0)
cnt = sorted(cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
mask = np.zeros((binary.shape[0],binary.shape[1]), np.uint8)
masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
plt.imshow(masked,cmap='gray')

########## instance segmentation
https://www.geeksforgeeks.org/python-opencv-connected-component-labeling-and-analysis/
https://www.youtube.com/watch?v=IGnIRp5dW_c

########## Fiber orientation
import math
# Leo la imagen monocroma
src = cv2.imread('cx_prueba_c1.tif')
src = cv2.morphologyEx(src, cv2.MORPH_OPEN, (3,3), iterations = 2)
src = cv2.erode(src, (5,5), iterations = 2)
#src = img_mono_equ

# Edge detection
dst = cv2.Canny(src, 50, 200, None, 3,True)
dst = cv2.dilate(dst, (5,5), iterations = 3)
dst = cv2.erode(dst, (5,5), iterations = 3)

#contours2, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#prueba = cv2.drawContours(dst,contours2,-1,(0,255,0),3)
from skimage import feature

def dominant_direction(img, sigma):
    """OrientationsJ's dominant direction"""
    axx, axy, ayy = feature.structure_tensor(
        img.astype(np.float32), sigma=sigma, mode="reflect"
    )
    dom_ori = np.arctan2(2 * axy.mean(), (ayy.mean() - axx.mean())) / 2
    return np.rad2deg(dom_ori)

img = cv2.imread('cx_sana.tif', cv2.IMREAD_GRAYSCALE)
orient = dominant_direction(img,1)
orient = dominant_direction(src[:,:,1],1) # cut 1 = -2 - cut 2 = -76. Distintos valores de Sigma no modifican resultado
#orient = (np.pi/180) * orient
#orient = np.pi/4
dkrgrowc2 = 5
#k45 =  np.array([[0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,1,0,0,0],
 #               [0,0,0,0,1,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]], np.uint8)
k_30 = np.array([[1,0,0,0,0], [0,1,1,0,0], [0,0,0,1,1]], np.uint8)
k30 = np.flipud(k_30)
k_45 = np.array([[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]], np.uint8)
k45 = np.flipud(k_45)
k_60 = np.array([[1,0,0], [0,1,0], [0,1,0], [0,0,1], [0,0,1]], np.uint8)
k60 = np.flipud(k_75)
n = np.flipud(k_45)
k0 = cv2.getStructuringElement(cv2.MORPH_RECT, (dkrgrowc2, 1))
k90 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, dkrgrowc2))

src2 = cv2.imread('2_Cx43_cut3_c2.tif')
kernel_noise_remove_c2 = np.ones((2,2),np.uint8)
thresh1cx = cv2.threshold(src2,254,255,cv2.THRESH_BINARY)[1]
ercx=cv2.morphologyEx(thresh1cx, cv2.MORPH_OPEN, kernel_noise_remove_c2)
gimgcx=cv2.dilate(ercx,k45,iterations = 5)


########################## filtro de ángulos

def angle_within_range(angle, dom_angle, range_size=20):
    """Check if an angle is within a range of another angle."""
    # Convert angles to positive values within the range [0, 180) degrees
    angle = (angle + 90) % 180 - 90
    dom_angle = (dom_angle + 90) % 180 - 90

    # Calculate the difference between the angles
    diff = abs(angle - dom_angle)

    # Wrap the difference around 180 degrees if it's larger than that
    if diff > 90:
        diff = 180 - diff

    # Check if the difference is within the range size
    return diff <= range_size


angle_within_range(-79, 80)



################################# nuevas imagenes emi (todas verdes)
img = cv2.imread('ejemplo.jpg')[:,:,1]
img_mono = cv2.imread('ejemplo.jpg',0) # lee la imagen monocroma

plt.figure(2)
plt.subplot(1,2,1),plt.imshow(img, cmap='gray'),plt.title("canal 1")
plt.subplot(1,2,2),plt.imshow(img_mono, cmap='gray'),plt.title("leida mono")
plt.show()

plt.figure(3)
img_equ = cv2.equalizeHist(img) # la ecualiza
plt.subplot(2,2,1),plt.imshow(img, cmap='gray'),plt.title("canal 1")
plt.subplot(2,2,2),plt.imshow(img_equ, cmap='gray'),plt.title("ecualizada")
plt.subplot(2,2,3),plt.hist(img.ravel(), 256,[0,256] ),plt.title("canal 1")
plt.subplot(2,2,4),plt.hist(img_equ.ravel(), 256,[0,256] ),plt.title("ecualizada")
plt.show()

tresh_cx = 240
tresh_cx_equ = tresh_cx + 0
cx = cv2.threshold(img,tresh_cx,255,cv2.THRESH_BINARY)[1]
cx_equ = cv2.threshold(img_equ,tresh_cx_equ,255,cv2.THRESH_BINARY)[1]
plt.figure(4)
plt.subplot(2,2,1),plt.imshow(img, cmap='gray'),plt.title("canal 1")
plt.subplot(2,2,2),plt.imshow(img_equ, cmap='gray'),plt.title("ecualizada")
plt.subplot(2,2,3),plt.imshow(cx, cmap='gray'),plt.title("canal 1")
plt.subplot(2,2,4),plt.imshow(cx_equ, cmap='gray'),plt.title("binarizada")
plt.show()


######################################################################### Mask manual
#Contours manual
nimg = 'cx_prueba'
mask_green = cv2.imread(nimg+'_Mm.tif',0)
contours, hierarchy = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#Contours automático
imgserca = cv2.imread(nimg+"_c1_binarized.tif",0)
imgcx = cv2.imread(nimg+"_c2_binarized.tif",0)
imgwga = cv2.imread(nimg+"_c3_binarized.tif",0)
img3comb = cv2.imread("blackback.tif") #fondo negro
img3comb[:,:,2]=cv2.add(imgwga,imgcx) # cv2.add combina las imagenes
img3comb[:,:,1]=cv2.add(imgserca,imgcx)
img3comb[:,:,0]=imgcx
up_mask_g=255
dwn_mask_g=250 
lower = np.array([0,dwn_mask_g,0])# [0,250,0]
upper = np.array([0,up_mask_g,0])# [0,255,0]
mask_green = cv2.inRange(img3comb, lower, upper)
contours2, hierarchy = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#Filtrado
boxc=[]
cardiomdet = 0
scale=0.227
ff_permin = 40
ff_areamin = 100
sf_lboxmin = 20
sf_lboxmax = 200
sf_rmin = 1
sf_wmax = 50
sf_wmin = 5
pad = 50
contoursfilt=[]
boxparts=[]		
for j in range(0,len(contours2)-1,1):        

# Computation of contour area and perimeter using OpenCV libraries
    area = cv2.contourArea(contours2[j])*scale**2
    perimeter = cv2.arcLength(contours2[j],True)*scale
    
# First filter:
    if (perimeter > ff_permin) & (area > ff_areamin): 

# Enclosing filtered contours in minimum area rectangles        
        rect = cv2.minAreaRect(contours2[j])
        ang = rect[2]
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        autowidth = rect[1][0]*scale
        autoheight = rect[1][1]*scale
        boxc.append(box)

# Axis lenght stimation   
        d01box=((box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2)**0.5 #(dy**2+dx**2)**0.5. Es el largo de una caja acostada
        d12box=((box[1][0]-box[2][0])**2+(box[1][1]-box[2][1])**2)**0.5 # Ancho de una caja acostada
# lenght and widths
        #Wbox = rect[1][0]*scale
        #Lbox = rect[1][1]*scale
        Lbox=np.maximum(d01box,d12box)*scale #largo
        Wbox=np.minimum(d01box,d12box)*scale #ancho

        newAngle=0.0
        if(autoheight > autowidth): newAngle = (-1)*ang
        else: newAngle = 90 - ang
        
        if(newAngle<0):ang = newAngle+90
        if(newAngle>0):ang = newAngle-90
        if(newAngle==0):ang = 90
        def angle_within_range(angle, dom_angle, range_size):
            """Check if an angle is within a range of another angle."""
            # Convert angles to positive values within the range [0, 180) degrees
            angle = (angle + 90) % 180 - 90
            dom_angle = (dom_angle + 90) % 180 - 90

            # Calculate the difference between the angles
            diff = abs(angle - dom_angle)

            # Wrap the difference around 180 degrees if it's larger than that
            if diff > 90:
                diff = 180 - diff

            # Check if the difference is within the range size
            return diff <= range_size
        def dominant_direction(img, sigma):
            """OrientationsJ's dominant direction"""
            axx, axy, ayy = feature.structure_tensor(
                img.astype(np.float32), sigma=sigma, mode="reflect"
            )
            dom_ori = np.arctan2(2 * axy.mean(), (ayy.mean() - axx.mean())) / 2
            return np.rad2deg(dom_ori)
        orient = dominant_direction(imgserca,1)
        if (angle_within_range(ang, orient, 30)==True):
    # Second filter applied        
            if (Lbox > sf_lboxmin) & (Lbox < sf_lboxmax) & (Lbox/Wbox > sf_rmin) & (Wbox < sf_wmax) & (Wbox > sf_wmin):
    #  CM counter       
                cardiomdet=cardiomdet+1
        
    #  Box partition coordinates            
                if np.minimum(d01box,d12box) == d01box:
                #if Wbox == d01box:
                    x0=box[0][0]
                    x1=box[1][0]
                    y0=box[0][1]
                    y1=box[1][1]
                    x2=box[2][0]
                    y2=box[2][1]
                    x3=box[3][0]
                    y3=box[3][1]
                    x4=int(round((x0+x1)/2))
                    y4=int(round((y0+y1)/2))
                    x5=int(round((x1+x2)/2))
                    y5=int(round((y1+y2)/2))
                    x6=int(round((x2+x3)/2))
                    y6=int(round((y2+y3)/2))
                    x7=int(round((x3+x0)/2))
                    y7=int(round((y3+y0)/2))
                    x8=int(round((x4+x0)/2))
                    y8=int(round((y4+y0)/2))
                    x9=int(round((x4+x1)/2))
                    y9=int(round((y4+y1)/2))        
                    x10=int(round((x5+x1)/2))
                    y10=int(round((y5+y1)/2))
                    x11=int(round((x5+x2)/2))
                    y11=int(round((y5+y2)/2))
                    x12=int(round((x6+x2)/2))
                    y12=int(round((y6+y2)/2))
                    x13=int(round((x6+x3)/2))
                    y13=int(round((y6+y3)/2))
                    x14=int(round((x7+x3)/2))
                    y14=int(round((y7+y3)/2))
                    x15=int(round((x7+x0)/2))
                    y15=int(round((y7+y0)/2))        
                else:
                    x0=box[1][0]
                    x1=box[2][0]
                    y0=box[1][1]
                    y1=box[2][1]
                    x2=box[3][0]
                    y2=box[3][1]
                    x3=box[0][0]
                    y3=box[0][1]
                    x4=int(round((x0+x1)/2))
                    y4=int(round((y0+y1)/2))
                    x5=int(round((x1+x2)/2))
                    y5=int(round((y1+y2)/2))
                    x6=int(round((x2+x3)/2))
                    y6=int(round((y2+y3)/2))
                    x7=int(round((x3+x0)/2))
                    y7=int(round((y3+y0)/2))
                    x8=int(round((x4+x0)/2))
                    y8=int(round((y4+y0)/2))
                    x9=int(round((x4+x1)/2))
                    y9=int(round((y4+y1)/2))        
                    x10=int(round((x5+x1)/2))
                    y10=int(round((y5+y1)/2))
                    x11=int(round((x5+x2)/2))
                    y11=int(round((y5+y2)/2))
                    x12=int(round((x6+x2)/2))
                    y12=int(round((y6+y2)/2))
                    x13=int(round((x6+x3)/2))
                    y13=int(round((y6+y3)/2))
                    x14=int(round((x7+x3)/2))
                    y14=int(round((y7+y3)/2))
                    x15=int(round((x7+x0)/2))
                    y15=int(round((y7+y0)/2)) 

    # Application of a padding to the rectangle 

                d13=((x0-x2)**2+(y0-y2)**2)**0.5
                d24=((x1-x3)**2+(y1-y3)**2)**0.5
                
                xn1=np.copy(x0)
                xn2=np.copy(x1)
                xn3=np.copy(x2)
                xn4=np.copy(x3)
                                    
                yn1=np.copy(y0)
                yn2=np.copy(y1)
                yn3=np.copy(y2)
                yn4=np.copy(y3)
                                                                   
                x0=xn1-int(round(pad*(xn3-xn1)/d13))
                x1=xn2-int(round(pad*(xn4-xn2)/d24))
                y0=yn1-int(round(pad*(yn3-yn1)/d13))
                y1=yn2-int(round(pad*(yn4-yn2)/d24))
                                    
                x2=xn3+int(round(pad*(xn3-xn1)/d13))
                y2=yn3+int(round(pad*(yn3-yn1)/d13))
                x3=xn4+int(round(pad*(xn4-xn2)/d24))
                y3=yn4+int(round(pad*(yn4-yn2)/d24))
                
    # Computing lenght of paded rectangle (extended in long axis direction)
                expbox1=((x0-x1)**2+(y0-y1)**2)**0.5
                expbox2=((x1-x2)**2+(y1-y2)**2)**0.5
                
    # Computing the coordinates range of the box             
                xmaxb=np.max((x0,x1,x2,x3))
                ymaxb=np.max((y0,y1,y2,y3))
                xminb=np.min((x0,x1,x2,x3))
                yminb=np.min((y0,y1,y2,y3))
                
    # Split CM box into 4 equal parts            
                rec1=np.array([[x0,y0],[x1,y1],[x10,y10],[x15,y15]])
                rec2=np.array([[x15,y15],[x10,y10],[x5,y5],[x7,y7]])
                rec3=np.array([[x7,y7],[x5,y5],[x11,y11],[x14,y14]])
                rec4=np.array([[x14,y14],[x11,y11],[x2,y2],[x3,y3]])  


    # Draw a background image 
                thresh, blackAndWhiteImage = cv2.threshold(imgserca, 0, 0, cv2.THRESH_BINARY)
                imgblack = blackAndWhiteImage
                

    # ch20220120
                contoursfilt.append(contours2[j])
                boxparts.append(rec1)
                boxparts.append(rec2)
                boxparts.append(rec3)
                boxparts.append(rec4)

contwidth= 4
for j in range(0,len(contoursfilt)-1,1):  
	cv2.drawContours(img3comb,[contoursfilt[j]],0,(128,128,0),contwidth) #ch20220120                    
for j in range(0,len(boxparts)-1,1):
    cv2.drawContours(img3comb,[boxparts[j]],0,(255,0,0),contwidth)       #ch20220120
plt.imshow(img3comb)







######################################################################### Manual & Automatic

#Manual
nimg = 'cx_prueba'
iman = cv2.imread(nimg+'_Mm.tif',0)

#Automático
imgserca = cv2.imread(nimg+"_c1_binarized.tif",0)
imgcx = cv2.imread(nimg+"_c2_binarized.tif",0)
imgwga = cv2.imread(nimg+"_c3_binarized.tif",0)
img3comb = cv2.imread("blackback.tif") #fondo negro
img3comb[:,:,2]=cv2.add(imgwga,imgcx) # cv2.add combina las imagenes
img3comb[:,:,1]=cv2.add(imgserca,imgcx)
img3comb[:,:,0]=imgcx
up_mask_g=255
dwn_mask_g=250 
lower = np.array([0,dwn_mask_g,0])# [0,250,0]
upper = np.array([0,up_mask_g,0])# [0,255,0]
imaut = cv2.inRange(img3comb, lower, upper)


retm,binthrman = cv2.threshold(iman,200,255,cv2.THRESH_BINARY)
retm,binthraut = cv2.threshold(imaut,200,255,cv2.THRESH_BINARY)
contman, hierarchy = cv2.findContours(binthrman, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contauto, hierarchy = cv2.findContours(binthraut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

lcnt=len(contman)
Lboxv=[]
Wboxv=[]
boxc=[]
Aboxv=[]
areamanv=[]
permanv=[]
scale=0.227
ff_permin = 40
ff_areamin = 100
sf_lboxmin = 20
sf_lboxmax = 200
sf_rmin = 1
sf_wmax = 50
sf_wmin = 5
pad = 50
amin=ff_areamin
amax=99999999.
lmin=sf_lboxmin
lmax=sf_lboxmax
wmin=sf_wmin
wmax=sf_wmax
permin=ff_permin
#################################################################### angulo

def dominant_direction(img, sigma):
    """OrientationsJ's dominant direction"""
    axx, axy, ayy = feature.structure_tensor(
        img.astype(np.float32), sigma=sigma, mode="reflect"
    )
    dom_ori = np.arctan2(2 * axy.mean(), (ayy.mean() - axx.mean())) / 2
    return np.rad2deg(dom_ori)
imgc1 = cv2.imread(nimg+'.tif', cv2.IMREAD_GRAYSCALE)
orient = dominant_direction(imgc1,1)
def angle_within_range(angle, dom_angle, range_size):
    """Check if an angle is within a range of another angle."""
    # Convert angles to positive values within the range [0, 180) degrees
    angle = (angle + 90) % 180 - 90
    dom_angle = (dom_angle + 90) % 180 - 90

    # Calculate the difference between the angles
    diff = abs(angle - dom_angle)

    # Wrap the difference around 180 degrees if it's larger than that
    if diff > 90:
        diff = 180 - diff

    # Check if the difference is within the range size
    return diff <= range_size
################################################################### angulo

thresh, blackAndWhiteImage = cv2.threshold(iman, 0, 0, cv2.THRESH_BINARY)
img_contours = blackAndWhiteImage
img_boxman= blackAndWhiteImage

for j in range(0,len(contman)-1,1):
    areaman = cv2.contourArea(contman[j])*scale**2 #área
    areamanv.append(areaman)
    perman = cv2.arcLength(contman[j],True)*scale #perimetro
    permanv.append(perman)
    rect = cv2.minAreaRect(contman[j])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    

    d01box=((box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2)**0.5
    d12box=((box[1][0]-box[2][0])**2+(box[1][1]-box[2][1])**2)**0.5
        
    Abox=d01box*d12box*scale**2
    Lbox=np.maximum(d01box,d12box)*scale
    Wbox=np.minimum(d01box,d12box)*scale
       
    if(Lbox > lmin) & (Lbox < lmax) & (areaman < amax) &(areaman> amin) &(perman>permin) &(Wbox<wmax)&(Wbox>wmin):
        Lboxv.append(Lbox)
        Aboxv.append(Abox)
        Wboxv.append(Wbox)
        boxc.append(box)
        cv2.drawContours(img_contours, [box], 0, (255,0,0), 2)
        cv2.drawContours(img_contours, contman[j], -1, (255, 255, 255), 2)
        cv2.drawContours(img_boxman, [box], 0, (255,0,0), 2)
Lboxva=[]
Wboxva=[]
boxca=[]
Aboxva=[]
areautova=[]
perautova=[]
thresh, blackAndWhiteImage = cv2.threshold(iman, 0, 0, cv2.THRESH_BINARY)
img_contaut = blackAndWhiteImage
img_boxauto= blackAndWhiteImage

for j in range(0,len(contauto)-1,1):
    areauto = cv2.contourArea(contauto[j])*scale**2
    areautova.append(areauto)
    perauto = cv2.arcLength(contauto[j],True)*scale
    perautova.append(perauto)
    recta = cv2.minAreaRect(contauto[j])
    boxa = cv2.boxPoints(recta)
    boxa = np.int0(boxa)
    rect = cv2.minAreaRect(contauto[j])
    ang = rect[2]

    d01boxa=((boxa[0][0]-boxa[1][0])**2+(boxa[0][1]-boxa[1][1])**2)**0.5
    d12boxa=((boxa[1][0]-boxa[2][0])**2+(boxa[1][1]-boxa[2][1])**2)**0.5
        
    Aboxa=d01boxa*d12boxa*scale**2
    Lboxa=np.maximum(d01boxa,d12boxa)*scale
    Wboxa=np.minimum(d01boxa,d12boxa)*scale

    

####################################################### angulo 2
    autowidth = rect[1][0]*scale
    autoheight = rect[1][1]*scale
    newAngle=0.0
    if(autoheight > autowidth): newAngle = (-1)*ang
    else: newAngle = 90 - ang
    
    if(newAngle<0):ang = newAngle+90
    if(newAngle>0):ang = newAngle-90
    if(newAngle==0):ang = 90
        
    if (angle_within_range(ang, orient, 30)==True):
####################################################### angulo 2
        
        if(Lboxa > lmin) & (Lboxa < lmax) & (areauto < amax) &(areauto> amin) &(perauto>permin) &(Wboxa<wmax)&(Wboxa>wmin):
            Lboxva.append(Lboxa)
            Wboxva.append(Wboxa)
            boxca.append(boxa)
            Aboxva.append(Aboxa)
            cv2.drawContours(img_contours, [boxa], 0, (0,0,255), 2)
            cv2.drawContours(img_contours, contauto[j], -1, (0, 255, 0), 1)
            cv2.drawContours(img_boxauto, [boxa], 0, (0,0,255), -1)



redb=[]
greenb=[]
relint=[]
modeval = 'ref'
mininters = 50
thresh, blackAndWhiteImage = cv2.threshold(iman, 0, 0, cv2.THRESH_BINARY) ## FALTABA ESTO EN EL CODIGO
for i in range(0,len(boxc)-1,1):
    imexample = np.copy(blackAndWhiteImage) # al parecer acá no llega vacía la imagen
    example=boxc[i];
    areaexample=cv2.contourArea(example)

    xmin=np.min((example[0][0],example[1][0],example[2][0],example[3][0]))
    xmax=np.max((example[0][0],example[1][0],example[2][0],example[3][0]))
    ymin=np.min((example[0][1],example[1][1],example[2][1],example[3][1]))
    ymax=np.max((example[0][1],example[1][1],example[2][1],example[3][1]))

    cv2.drawContours(imexample, [example], 0, (255,0,0), -1)

    intersec=[]
    for j in range(0,len(boxca)-1,1):
        imloop = np.copy(blackAndWhiteImage)
        boxcap=boxca[j]
        areaboxcap=cv2.contourArea(boxcap)
        
        xmin2=np.min((boxcap[0][0],boxcap[1][0],boxcap[2][0],boxcap[3][0]))
        xmax2=np.max((boxcap[0][0],boxcap[1][0],boxcap[2][0],boxcap[3][0]))
        ymin2=np.min((boxcap[0][1],boxcap[1][1],boxcap[2][1],boxcap[3][1]))
        ymax2=np.max((boxcap[0][1],boxcap[1][1],boxcap[2][1],boxcap[3][1]))

        xmin3=np.min((xmin,xmin2))
        xmax3=np.max((xmax,xmax2))
        ymin3=np.min((ymin,ymin2))
        ymax3=np.max((ymax,ymax2))

        autsubr3=imloop[ymin3:ymax3,xmin3:xmax3]
        mansubr3=imexample[ymin3:ymax3,xmin3:xmax3]
        mansubr=imexample[ymin:ymax,xmin:xmax]
        autosubr2=imloop[ymin2:ymax2,xmin2:xmax2]
        
        cv2.drawContours(imloop, [boxcap], 0, (255,0,0), -1)
               
        intsec=cv2.bitwise_and(autsubr3,mansubr3)
        area1=np.count_nonzero(cv2.bitwise_and(mansubr, mansubr))
        area2=np.count_nonzero(cv2.bitwise_and(autosubr2,autosubr2))
        if(modeval=='max'):
            intersec.append(np.count_nonzero(intsec)/np.max((area1,area2))*100) #Calcula un índice de intersección dividiendo el número de píxeles no nulos en la imagen de intersección (intsec) por el área correspondiente (area1 o area2). El índice de intersección se multiplica por 100 para obtener un valor porcentual.
        else:
            intersec.append(np.count_nonzero(intsec)/(area2+0.00001)*100)
            
    boxmaxint_index=np.argmax(intersec)
    boxmaxint=boxca[boxmaxint_index]

    if(np.max(intersec)>mininters): # TOMA COMO INTERSECCION MINIMA EL 50%  ????
        relint.append(np.max(intersec))
        greenb.append(boxmaxint)
        box=boxmaxint
        d01box=((box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2)**0.5
        d12box=((box[1][0]-box[2][0])**2+(box[1][1]-box[2][1])**2)**0.5
        if np.minimum(d01box,d12box) == d01box:
        #if Wbox == d01box:
            x0=box[0][0]
            x1=box[1][0]
            y0=box[0][1]
            y1=box[1][1]
            x2=box[2][0]
            y2=box[2][1]
            x3=box[3][0]
            y3=box[3][1]
            x4=int(round((x0+x1)/2))
            y4=int(round((y0+y1)/2))
            x5=int(round((x1+x2)/2))
            y5=int(round((y1+y2)/2))
            x6=int(round((x2+x3)/2))
            y6=int(round((y2+y3)/2))
            x7=int(round((x3+x0)/2))
            y7=int(round((y3+y0)/2))
            x8=int(round((x4+x0)/2))
            y8=int(round((y4+y0)/2))
            x9=int(round((x4+x1)/2))
            y9=int(round((y4+y1)/2))        
            x10=int(round((x5+x1)/2))
            y10=int(round((y5+y1)/2))
            x11=int(round((x5+x2)/2))
            y11=int(round((y5+y2)/2))
            x12=int(round((x6+x2)/2))
            y12=int(round((y6+y2)/2))
            x13=int(round((x6+x3)/2))
            y13=int(round((y6+y3)/2))
            x14=int(round((x7+x3)/2))
            y14=int(round((y7+y3)/2))
            x15=int(round((x7+x0)/2))
            y15=int(round((y7+y0)/2))        
        else:
            x0=box[1][0]
            x1=box[2][0]
            y0=box[1][1]
            y1=box[2][1]
            x2=box[3][0]
            y2=box[3][1]
            x3=box[0][0]
            y3=box[0][1]
            x4=int(round((x0+x1)/2))
            y4=int(round((y0+y1)/2))
            x5=int(round((x1+x2)/2))
            y5=int(round((y1+y2)/2))
            x6=int(round((x2+x3)/2))
            y6=int(round((y2+y3)/2))
            x7=int(round((x3+x0)/2))
            y7=int(round((y3+y0)/2))
            x8=int(round((x4+x0)/2))
            y8=int(round((y4+y0)/2))
            x9=int(round((x4+x1)/2))
            y9=int(round((y4+y1)/2))        
            x10=int(round((x5+x1)/2))
            y10=int(round((y5+y1)/2))
            x11=int(round((x5+x2)/2))
            y11=int(round((y5+y2)/2))
            x12=int(round((x6+x2)/2))
            y12=int(round((y6+y2)/2))
            x13=int(round((x6+x3)/2))
            y13=int(round((y6+y3)/2))
            x14=int(round((x7+x3)/2))
            y14=int(round((y7+y3)/2))
            x15=int(round((x7+x0)/2))
            y15=int(round((y7+y0)/2)) 

# Application of a padding to the rectangle 

        d13=((x0-x2)**2+(y0-y2)**2)**0.5
        d24=((x1-x3)**2+(y1-y3)**2)**0.5
        
        xn1=np.copy(x0)
        xn2=np.copy(x1)
        xn3=np.copy(x2)
        xn4=np.copy(x3)
                            
        yn1=np.copy(y0)
        yn2=np.copy(y1)
        yn3=np.copy(y2)
        yn4=np.copy(y3)
                                                           
        x0=xn1-int(round(pad*(xn3-xn1)/d13))
        x1=xn2-int(round(pad*(xn4-xn2)/d24))
        y0=yn1-int(round(pad*(yn3-yn1)/d13))
        y1=yn2-int(round(pad*(yn4-yn2)/d24))
                            
        x2=xn3+int(round(pad*(xn3-xn1)/d13))
        y2=yn3+int(round(pad*(yn3-yn1)/d13))
        x3=xn4+int(round(pad*(xn4-xn2)/d24))
        y3=yn4+int(round(pad*(yn4-yn2)/d24))
        
# Computing lenght of paded rectangle (extended in long axis direction)
        expbox1=((x0-x1)**2+(y0-y1)**2)**0.5
        expbox2=((x1-x2)**2+(y1-y2)**2)**0.5
        
# # Computing the coordinates range of the box             
#         xmaxb=np.max((x0,x1,x2,x3))
#         ymaxb=np.max((y0,y1,y2,y3))
#         xminb=np.min((x0,x1,x2,x3))
#         yminb=np.min((y0,y1,y2,y3))
        
# # Split CM box into 4 equal parts            
#         rec1=np.array([[x0,y0],[x1,y1],[x10,y10],[x15,y15]])
#         rec2=np.array([[x15,y15],[x10,y10],[x5,y5],[x7,y7]])
#         rec3=np.array([[x7,y7],[x5,y5],[x11,y11],[x14,y14]])
#         rec4=np.array([[x14,y14],[x11,y11],[x2,y2],[x3,y3]])  


# # Draw a background image 
#         imgblack = blackAndWhiteImage
        
# # Compute an intersection between c2 channel and partitioned rectangles
#         r0fig=cv2.drawContours(imgblack,[rec1],0,(255,255,255),-1)
#         int0=cv2.bitwise_and(r0fig[yminb:ymaxb,xminb:xmaxb],imgcxb2[yminb:ymaxb,xminb:xmaxb])
#         nwp0=np.count_nonzero(int0)
#         imgblack = blackAndWhiteImage
#         r1fig=cv2.drawContours(imgblack,[rec2],0,(255,255,255),-1)
#         int1=cv2.bitwise_and(r1fig[yminb:ymaxb,xminb:xmaxb],imgcxb2[yminb:ymaxb,xminb:xmaxb])
#         nwp1=np.count_nonzero(int1)
#         imgblack = blackAndWhiteImage
#         r2fig=cv2.drawContours(imgblack,[rec3],0,(255,255,255),-1)
#         int2=cv2.bitwise_and(r2fig[yminb:ymaxb,xminb:xmaxb],imgcxb2[yminb:ymaxb,xminb:xmaxb])
#         nwp2=np.count_nonzero(int2)
#         imgblack = blackAndWhiteImage
#         r3fig=cv2.drawContours(imgblack,[rec4],0,(255,255,255),-1)
#         int3=cv2.bitwise_and(r3fig[yminb:ymaxb,xminb:xmaxb],imgcxb2[yminb:ymaxb,xminb:xmaxb])
#         nwp3=np.count_nonzero(int3)
        
# # Compute the proportions of c2 relative to each compartment          
#         p0=nwp0/(nwp0+nwp1+nwp2+nwp3+0.001)*100
#         p1=nwp1/(nwp0+nwp1+nwp2+nwp3+0.001)*100
#         p2=nwp2/(nwp0+nwp1+nwp2+nwp3+0.001)*100
#         p3=nwp3/(nwp0+nwp1+nwp2+nwp3+0.001)*100  
        
# Compute the lenghts and widths of CMs (lenghts as average of extended and non extended rectangle)
        Lbox0=np.maximum(d01box,d12box)*scale
        #Lbox0=Lbox
        Lbox1=np.maximum(expbox1,expbox2)*scale
        Lbox=(Lbox0+Lbox1)*0.5
        Wbox=np.minimum(d01box,d12box)*scale
        Abox=Lbox*Wbox
        #f4.write(str(round(Lbox,2))+','+str(round(Wbox,2))+','+str(round((p1+p2)*100/(p0+p3+p1+p2+0.001),2))+','+str(round(Abox,2))+','+str(round(area,2))+','+str(round(perimeter,2))+','+str(round(np.max(intersec)))+'\n')

        box=example
        d01box=((box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2)**0.5
        d12box=((box[1][0]-box[2][0])**2+(box[1][1]-box[2][1])**2)**0.5
        if np.minimum(d01box,d12box) == d01box:
        #if Wbox == d01box:
            x0=box[0][0]
            x1=box[1][0]
            y0=box[0][1]
            y1=box[1][1]
            x2=box[2][0]
            y2=box[2][1]
            x3=box[3][0]
            y3=box[3][1]
            x4=int(round((x0+x1)/2))
            y4=int(round((y0+y1)/2))
            x5=int(round((x1+x2)/2))
            y5=int(round((y1+y2)/2))
            x6=int(round((x2+x3)/2))
            y6=int(round((y2+y3)/2))
            x7=int(round((x3+x0)/2))
            y7=int(round((y3+y0)/2))
            x8=int(round((x4+x0)/2))
            y8=int(round((y4+y0)/2))
            x9=int(round((x4+x1)/2))
            y9=int(round((y4+y1)/2))        
            x10=int(round((x5+x1)/2))
            y10=int(round((y5+y1)/2))
            x11=int(round((x5+x2)/2))
            y11=int(round((y5+y2)/2))
            x12=int(round((x6+x2)/2))
            y12=int(round((y6+y2)/2))
            x13=int(round((x6+x3)/2))
            y13=int(round((y6+y3)/2))
            x14=int(round((x7+x3)/2))
            y14=int(round((y7+y3)/2))
            x15=int(round((x7+x0)/2))
            y15=int(round((y7+y0)/2))        
        else:
            x0=box[1][0]
            x1=box[2][0]
            y0=box[1][1]
            y1=box[2][1]
            x2=box[3][0]
            y2=box[3][1]
            x3=box[0][0]
            y3=box[0][1]
            x4=int(round((x0+x1)/2))
            y4=int(round((y0+y1)/2))
            x5=int(round((x1+x2)/2))
            y5=int(round((y1+y2)/2))
            x6=int(round((x2+x3)/2))
            y6=int(round((y2+y3)/2))
            x7=int(round((x3+x0)/2))
            y7=int(round((y3+y0)/2))
            x8=int(round((x4+x0)/2))
            y8=int(round((y4+y0)/2))
            x9=int(round((x4+x1)/2))
            y9=int(round((y4+y1)/2))        
            x10=int(round((x5+x1)/2))
            y10=int(round((y5+y1)/2))
            x11=int(round((x5+x2)/2))
            y11=int(round((y5+y2)/2))
            x12=int(round((x6+x2)/2))
            y12=int(round((y6+y2)/2))
            x13=int(round((x6+x3)/2))
            y13=int(round((y6+y3)/2))
            x14=int(round((x7+x3)/2))
            y14=int(round((y7+y3)/2))
            x15=int(round((x7+x0)/2))
            y15=int(round((y7+y0)/2)) 

# Application of a padding to the rectangle 

        d13=((x0-x2)**2+(y0-y2)**2)**0.5
        d24=((x1-x3)**2+(y1-y3)**2)**0.5
        
        xn1=np.copy(x0)
        xn2=np.copy(x1)
        xn3=np.copy(x2)
        xn4=np.copy(x3)
                            
        yn1=np.copy(y0)
        yn2=np.copy(y1)
        yn3=np.copy(y2)
        yn4=np.copy(y3)
                                                           
        x0=xn1-int(round(pad*(xn3-xn1)/d13))
        x1=xn2-int(round(pad*(xn4-xn2)/d24))
        y0=yn1-int(round(pad*(yn3-yn1)/d13))
        y1=yn2-int(round(pad*(yn4-yn2)/d24))
                            
        x2=xn3+int(round(pad*(xn3-xn1)/d13))
        y2=yn3+int(round(pad*(yn3-yn1)/d13))
        x3=xn4+int(round(pad*(xn4-xn2)/d24))
        y3=yn4+int(round(pad*(yn4-yn2)/d24))
        
# Computing lenght of paded rectangle (extended in long axis direction)
        expbox1=((x0-x1)**2+(y0-y1)**2)**0.5
        expbox2=((x1-x2)**2+(y1-y2)**2)**0.5
        
# Computing the coordinates range of the box             
        xmaxb=np.max((x0,x1,x2,x3))
        ymaxb=np.max((y0,y1,y2,y3))
        xminb=np.min((x0,x1,x2,x3))
        yminb=np.min((y0,y1,y2,y3))
        
# Split CM box into 4 equal parts            
        rec1=np.array([[x0,y0],[x1,y1],[x10,y10],[x15,y15]])
        rec2=np.array([[x15,y15],[x10,y10],[x5,y5],[x7,y7]])
        rec3=np.array([[x7,y7],[x5,y5],[x11,y11],[x14,y14]])
        rec4=np.array([[x14,y14],[x11,y11],[x2,y2],[x3,y3]])  


# Draw a background image 
        imgblack = np.copy(blackAndWhiteImage)
        
# # Compute an intersection between c2 channel and partitioned rectangles
#         r0fig=cv2.drawContours(imgblack,[rec1],0,(255,255,255),-1)
#         int0=cv2.bitwise_and(r0fig[yminb:ymaxb,xminb:xmaxb],imgcxb2[yminb:ymaxb,xminb:xmaxb])
#         nwp0=np.count_nonzero(int0)
#         imgblack = blackAndWhiteImage
#         r1fig=cv2.drawContours(imgblack,[rec2],0,(255,255,255),-1)
#         int1=cv2.bitwise_and(r1fig[yminb:ymaxb,xminb:xmaxb],imgcxb2[yminb:ymaxb,xminb:xmaxb])
#         nwp1=np.count_nonzero(int1)
#         imgblack = blackAndWhiteImage
#         r2fig=cv2.drawContours(imgblack,[rec3],0,(255,255,255),-1)
#         int2=cv2.bitwise_and(r2fig[yminb:ymaxb,xminb:xmaxb],imgcxb2[yminb:ymaxb,xminb:xmaxb])
#         nwp2=np.count_nonzero(int2)
#         imgblack = blackAndWhiteImage
#         r3fig=cv2.drawContours(imgblack,[rec4],0,(255,255,255),-1)
#         int3=cv2.bitwise_and(r3fig[yminb:ymaxb,xminb:xmaxb],imgcxb2[yminb:ymaxb,xminb:xmaxb])
#         nwp3=np.count_nonzero(int3)
        
# # Compute the proportions of c2 relative to each compartment          
#         p0=nwp0/(nwp0+nwp1+nwp2+nwp3+0.001)*100
#         p1=nwp1/(nwp0+nwp1+nwp2+nwp3+0.001)*100
#         p2=nwp2/(nwp0+nwp1+nwp2+nwp3+0.001)*100
#         p3=nwp3/(nwp0+nwp1+nwp2+nwp3+0.001)*100  
        
# Compute the lenghts and widths of CMs (lenghts as average of extended and non extended rectangle)
        Lbox0=np.maximum(d01box,d12box)*scale
        #Lbox0=Lbox
        Lbox1=np.maximum(expbox1,expbox2)*scale
        Lbox=(Lbox0+Lbox1)*0.5
        Wbox=np.minimum(d01box,d12box)*scale
        Abox=Lbox*Wbox
        #f5.write(str(round(Lbox,2))+','+str(round(Wbox,2))+','+str(round((p1+p2)*100/(p0+p3+p1+p2+0.001),2))+','+str(round(Abox,2))+','+str(round(area,2))+','+str(round(perimeter,2))+','+str(round(np.max(intersec)))+'\n')                                 
    else:
        redb.append(boxmaxint)
        
# for j in range(0,len(greenb),1):
#     cv2.drawContours(img_boxman, [greenb[j]], 0, (0,0,255), 2)

# if(scalebars=='y'):
#     cv2.drawContours(img_boxman,[scbar],-1,(255,255,255),-1)      
# cv2.imwrite("out_quantif_"+dirstr+"/"+idim+"_"+distr+"_MaxAnd"+outf,img_boxman)




# COMPUTING PERCENTILE PLOTS
percentlist=[]
xdata=[]
for i in range(0,100,1):
    percentlist.append(np.percentile(relint,i))
    xdata.append(i)
    #f.write(str(i)+","+str(round(np.percentile(relint,i),1))+'\n')
auc = np.trapz(percentlist,xdata)
auc/100
 
plt.figure(figsize=(15, 10))
plt.plot(xdata,percentlist);
# Add title and axis names

plt.ylabel('Pc(k)', fontsize=18, family="serif")
plt.xlabel('k' , fontsize=18, family="serif")
plt.text(90, 50, "AUC="+str(round(auc/10000,2)), family="serif", fontsize=18,bbox=dict(facecolor='red', alpha=0.5));
plt.tick_params(direction='in', length=5, width=3, grid_alpha=0.5, labelsize=18)

# ME DA AUC = 0,84!!!!!!!!!!!!!!¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡

######################################################################### IMAGEN 9

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
#nimg = '/home/dgattari/Descargas/9_Cx43' # para UTN
nimg = '9_Cx43'
Image.MAX_IMAGE_PIXELS = None

im = np.array(Image.open(nimg+'.tif'), dtype=np.uint8)
img = np.array(Image.open('cx_prueba.tif'), dtype=np.uint8)

#obtiene dimensiones de img
#h, w, _ = img.shape
h , w = 10000,13000
#define tamaño de la ventana
ventana = (h, w)
#cantidad de imágenes
x = 100
#posicion inicial de la ventana
pos_x = 0
pos_y = 0
#bucle para el ventaneo y generar las nuevas imagenes
for i in range(x):
    # Definir la posición final de la ventana
    fin_x = pos_x + w
    fin_y = pos_y + h
    
    # Seleccionar la sección de la imagen "im" que corresponde a la ventana
    ventana_im = im[pos_y:fin_y, pos_x:fin_x]
    
    # Guardar la imagen resultante con el nombre "im1", "im2", "im3", etc.
    nombre = 'Ventaneo_im9/im' + str(i+1) + '.tif'
    cv2.imwrite(nombre, ventana_im)
    
    # Actualizar la posición inicial de la ventana para la siguiente iteración
    pos_x += w
    
    # Si se alcanza el final de la imagen en el eje X, se reinicia en la posición inicial en X
    if pos_x + w > im.shape[1]:
        pos_x = 0
        pos_y += h
    
    # Si se alcanza el final de la imagen en el eje Y, se sale del bucle
    if pos_y + h > im.shape[0]:
        break







corr1 = 0
corr2 = 0
inifil = 23903 + corr1
finfil = 27000 + corr1
inicol = 31096 + corr2
fincol = 36010 + corr2


cut1 = im[inifil:finfil,inicol:fincol,:]
plt.imshow(cut1)
# col aumenta hacia la derecha y (se supone) fila hacia abajo

cv2.imwrite('cx_trans1.tif', cut1)

#--------------------------------------------

corr1 = -400
corr2 = 3000
inifil = 23903 + corr1
finfil = 27000 + corr1
inicol = 25636 + corr2
fincol = 30550 + corr2


cut1 = im[inifil:finfil,inicol:fincol,:]
plt.imshow(cut1)
# col aumenta hacia la derecha y (se supone) fila hacia abajo

cv2.imwrite('cx_trans2.tif', cut1)

#--------------------------------------------

corr1 = -900
corr2 = 3000
inifil = 23903 + corr1
finfil = 27000 + corr1
inicol = 25636 + corr2
fincol = 30550 + corr2


cut1 = im[inifil:finfil,inicol:fincol,:]
plt.imshow(cut1)
# col aumenta hacia la derecha y (se supone) fila hacia abajo

cv2.imwrite('cx_trans3.tif', cut1)

#--------------------------------------------

corr1 = -1600
corr2 = 2000
inifil = 23903 + corr1
finfil = 27000 + corr1
inicol = 25636 + corr2
fincol = 30550 + corr2


cut1 = im[inifil:finfil,inicol:fincol,:]
plt.imshow(cut1)
# col aumenta hacia la derecha y (se supone) fila hacia abajo

cv2.imwrite('cx_trans4.tif', cut1)

#--------------------------------------------

corr1 = -2000
corr2 = 1500
inifil = 23903 + corr1
finfil = 27000 + corr1
inicol = 25636 + corr2
fincol = 30550 + corr2


cut1 = im[inifil:finfil,inicol:fincol,:]
plt.imshow(cut1)
# col aumenta hacia la derecha y (se supone) fila hacia abajo

cv2.imwrite('cx_trans5.tif', cut1)

#--------------------------------------------

corr1 = -2000
corr2 = 3500
inifil = 23903 + corr1
finfil = 27000 + corr1
inicol = 25636 + corr2
fincol = 30550 + corr2


cut1 = im[inifil:finfil,inicol:fincol,:]
plt.imshow(cut1)
# col aumenta hacia la derecha y (se supone) fila hacia abajo

cv2.imwrite('cx_trans6.tif', cut1)

#--------------------------------------------

corr1 = 0
corr2 = 0
inifil = 18153 + corr1
finfil = 21250 + corr1
inicol = 31486 + corr2
fincol = 36400 + corr2


cut1 = im[inifil:finfil,inicol:fincol,:]
plt.imshow(cut1)
# col aumenta hacia la derecha y (se supone) fila hacia abajo

cv2.imwrite('cx_enf1.tif', cut1)

#--------------------------------------------

corr1 = 0
corr2 = 0
inifil = 15000 + corr1
finfil = 18097 + corr1
inicol = 23686 + corr2
fincol = 28600 + corr2


cut1 = im[inifil:finfil,inicol:fincol,:]
plt.imshow(cut1)
# col aumenta hacia la derecha y (se supone) fila hacia abajo

cv2.imwrite('cx_enf2.tif', cut1)

#--------------------------------------------

corr1 = 1500
corr2 = 800
inifil = 10203 + corr1
finfil = 13300 + corr1
inicol = 11986 + corr2
fincol = 16900 + corr2


cut1 = im[inifil:finfil,inicol:fincol,:]
plt.imshow(cut1)
# col aumenta hacia la derecha y (se supone) fila hacia abajo

cv2.imwrite('cx_enf3.tif', cut1)

#--------------------------------------------

corr1 = 2000
corr2 = 0
inifil = 10203 + corr1
finfil = 13300 + corr1
inicol = 11986 + corr2
fincol = 16900 + corr2


cut1 = im[inifil:finfil,inicol:fincol,:]
plt.imshow(cut1)
# col aumenta hacia la derecha y (se supone) fila hacia abajo

cv2.imwrite('cx_enf4.tif', cut1)



######################################################################################## GRID SEARCH IMAGENES

from sklearn.model_selection import GridSearchCV
import numpy as np
image = cv2.imread('cx_trans2.tif',0)
folder = 'test_grid_trans2'
w = np.shape(image)[1]

# Define la función que deseas optimizar
def funcion(im, width, offset):
    neq = im # lee la imagen monocroma
    img=cv2.equalizeHist(neq)
    scale=0.227
    backg = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,width,offset)
    tiss = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,width,offset)    
    thresh, blackAndWhiteImage = cv2.threshold(img, 0, 0, cv2.THRESH_BINARY) # imagen completamente en negro
    cv2.imwrite(folder+"/blackback.tif",blackAndWhiteImage) # guarda la imagen en la carpeta
    imgback = cv2.imread(folder+"/blackback.tif")
    imgfor = cv2.imread(folder+"/blackback.tif")
    imgtis = cv2.imread(folder+"/blackback.tif")
    imgback[:,:,2]=tiss # guarda el fondo (medio extracelular) en el canal rojo, los demás canales valen 0
    imgback[:,:,1]=0
    imgback[:,:,0]=0
    #imgfor[:,:,2]=foreg # guarda la conexina en los 3 canales (quedan blancas)
    #imgfor[:,:,1]=foreg
    #imgfor[:,:,0]=foreg
    imgtis[:,:,2]=0
    imgtis[:,:,1]=backg # guarda el medio intracelular en el canal verde, los demás valen 0
    imgtis[:,:,0]=0
    cv2.imwrite(folder+"/w"+str(width)+"off"+str(offset)+"_c3.tif", imgback) # guarda las 3 imágenes de los 3 canales en la carpeta
    cv2.imwrite(folder+"/w"+str(width)+"off"+str(offset)+"_c1.tif", imgtis)
      
    # Haz algo con los valores de "width" y "offset"
    pass

# Define la lista de valores para "width"
wini = np.round(w*0.05)
wfin = np.round(w*0.20)
widths = list(np.uint(np.round(np.linspace(wini, wfin, 20))))
for i in range(len(widths)):
    if widths[i] % 2 == 0:  # Verificar si el número es par
        widths[i] += 1  # Incrementar el número en 1 para hacerlo impar

# Define la lista de valores para "offset"
offsets = list(np.uint(np.linspace(0, 70, 10)))

# Ejecuta la función para todas las combinaciones de valores de "width" y "offset"
for width in widths:
    for offset in offsets:
        funcion(image, width, offset)




############## Filtro porcentaje pixeles

# Filtrar imágenes que no tienen proporción de pixeles adecuada
import os
import glob
import pandas as pd
import seaborn as sns

# load the training dataset
train_path = "test_grid2"
train_names = os.listdir(train_path)

# empty list to hold feature vectors and train labels
perc = []

def calc_percentage(image):
        perc = np.sum(image)/np.size(image)

        return perc

# Create dictionaries to store the feature values
feature_dict = {
    'perc': {}
}

# loop over the training dataset
for train_name in train_names:
        cur_path = train_path + "/" + train_name
        cur_label = train_name
        #i = 1
        for file in glob.glob(cur_path):
                # read the training image
                image = cv2.imread(file)

                # convert the image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = np.uint(gray/np.max(gray))
                # extract haralick texture from the image
                image_perc = calc_percentage(gray)

                # append the feature vector and label
                perc.append(image_perc)
                
                # Extract width and offset from the file name
                _, filename = os.path.split(file)
                w_start = filename.find("w") + 1
                off_start = filename.find("off") + 3
                off_end = filename.find("_c1")
                width = int(filename[w_start:off_start-3])
                offset = int(filename[off_start:off_end])
        
                # Store feature values in the corresponding dictionaries
                feature_dict['perc'].setdefault(width, {})[offset] = image_perc




# Create dataframes from the dictionaries
df_perc = pd.DataFrame.from_dict(feature_dict['perc'], orient='index')
# Ordenar las columnas de menor a mayor
df_perc = df_perc.sort_index(axis=1)
# Ordenar las filas de menor a mayor
df_perc = df_perc.sort_index(axis=0)


# Crear el heatmap
plt.figure(figsize=(10, 8))
#sns.heatmap(df, cmap="YlGnBu")
sns.heatmap(df_perc, cmap="rocket")
#sns.heatmap(df, cmap="vlag")

# Mostrar el heatmap
plt.xlabel("Offset", fontsize=14)
plt.ylabel("Ancho", fontsize=14)
plt.title("Heatmap de porcentaje de pixeles blancos", fontsize=16)

# Mostrar el heatmap
plt.show()



######################################################################################## GRID SEARCH PARAMETERS

from skimage.feature import graycomatrix, graycoprops
import os
import glob
import pandas as pd
import seaborn as sns

# load the training dataset
train_path = "test_grid_enf1"
train_names = os.listdir(train_path)

# empty list to hold feature vectors and train labels
cont = []
diss = []
homo = []
ener = []
corr = []
ASM = []
train_labels = []
glcm = []

def extract_features(image):
        glcm = graycomatrix(image, distances=[5], angles=[0], levels=2,
                        symmetric=True, normed=True)
        #glcm = graycomatrix(image, distances=[1,2,3], angles=[0, 45, 90, 135], levels=2)
        cont  =graycoprops(glcm, 'contrast')[0, 0]
        diss  =graycoprops(glcm, 'dissimilarity')[0, 0]
        homo  =graycoprops(glcm, 'homogeneity')[0, 0]
        ener  =graycoprops(glcm, 'energy')[0, 0]
        corr =graycoprops(glcm, 'correlation')[0, 0]
        ASM =graycoprops(glcm, 'ASM')[0, 0]

        return cont,diss,homo,ener,corr,ASM,glcm

# Create dictionaries to store the feature values
feature_dict = {
    'cont': {},
    'diss': {},
    'homo': {},
    'ener': {},
    'corr': {},
    'ASM': {}
}

# loop over the training dataset
print ("[STATUS] Started extracting haralick textures..")
i=1
for train_name in train_names:
        cur_path = train_path + "/" + train_name
        cur_label = train_name
        #i = 1
        for file in glob.glob(cur_path):
                print ("Processing Image - {} in {}".format(i, np.size(train_names)))
                # read the training image
                image = cv2.imread(file)

                # convert the image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = np.uint(gray/np.max(gray))
                # extract haralick texture from the image
                features = extract_features(gray)

                # append the feature vector and label
                cont.append(features[0])
                diss.append(features[1])
                homo.append(features[2])
                ener.append(features[3])
                corr.append(features[4])
                ASM.append(features[5])
                train_labels.append(cur_label)
                glcm.append(features[6])
                
                # Extract width and offset from the file name
                _, filename = os.path.split(file)
                w_start = filename.find("w") + 1
                off_start = filename.find("off") + 3
                off_end = filename.find("_c1")
                width = int(filename[w_start:off_start-3])
                offset = int(filename[off_start:off_end])
        
                # Store feature values in the corresponding dictionaries
                feature_dict['cont'].setdefault(width, {})[offset] = features[0]
                feature_dict['diss'].setdefault(width, {})[offset] = features[1]
                feature_dict['homo'].setdefault(width, {})[offset] = features[2]
                feature_dict['ener'].setdefault(width, {})[offset] = features[3]
                feature_dict['corr'].setdefault(width, {})[offset] = features[4]
                feature_dict['ASM'].setdefault(width, {})[offset] = features[5]


                # show loop update
                i += 1




# Create dataframes from the dictionaries
df_cont = pd.DataFrame.from_dict(feature_dict['cont'], orient='index')
df_diss = pd.DataFrame.from_dict(feature_dict['diss'], orient='index')
df_homo = pd.DataFrame.from_dict(feature_dict['homo'], orient='index')
df_ener = pd.DataFrame.from_dict(feature_dict['ener'], orient='index')
df_corr = pd.DataFrame.from_dict(feature_dict['corr'], orient='index')
df_ASM = pd.DataFrame.from_dict(feature_dict['ASM'], orient='index')

# Ordenar las columnas de menor a mayor
df_cont = df_cont.sort_index(axis=1)
df_diss = df_diss.sort_index(axis=1)
df_homo = df_homo.sort_index(axis=1)
df_ener = df_ener.sort_index(axis=1)
df_corr = df_corr.sort_index(axis=1)
df_ASM = df_ASM.sort_index(axis=1)

# Ordenar las filas de menor a mayor
df_cont = df_cont.sort_index(axis=0)
df_diss = df_diss.sort_index(axis=0)
df_homo = df_homo.sort_index(axis=0)
df_ener = df_ener.sort_index(axis=0)
df_corr = df_corr.sort_index(axis=0)
df_ASM = df_ASM.sort_index(axis=0)

# Lista de dataframes
dataframes = [df_cont, df_diss, df_homo, df_ener, df_corr, df_ASM]
titles = ['Contraste', 'Dissimilarity', 'Homogeneidad', 'Energía', 'Correlación', 'ASM']

# Aplicar formato a cada dataframe
for i, df in enumerate(dataframes):
    # Crear el heatmap
    plt.figure(figsize=(10, 8))
    #sns.heatmap(df, cmap="YlGnBu")
    sns.heatmap(df, cmap="rocket")
    #sns.heatmap(df, cmap="vlag")
    
    # Mostrar el heatmap
    plt.xlabel("Offset", fontsize=14)
    plt.ylabel("Ancho", fontsize=14)
    plt.title("Heatmap de {}".format(titles[i]), fontsize=16)

    # Mostrar el heatmap
    plt.show()
        
################################################## SUPERFICIES
from mpl_toolkits.mplot3d import Axes3D

# Lista de dataframes
dataframes = [df_cont, df_diss, df_homo, df_ener, df_corr, df_ASM]
titles = ['Contraste', 'Dissimilarity', 'Homogeneidad', 'Energía', 'Correlación', 'ASM']


# Dibujar superficie para cada dataframe
for i, df in enumerate(dataframes):
    # Crear las coordenadas x, y y z
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(df.shape[1])
    y = np.arange(df.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = df.values

    # Dibujar la superficie
    ax.plot_surface(X, Y, Z, cmap='rocket')

    # Configurar los títulos de los ejes
    ax.set_xlabel('Offset')
    ax.set_ylabel('Ancho')
    ax.set_zlabel(titles[i])

    # Agregar título al gráfico
    ax.set_title("Superficie de {}".format(titles[i]))

    # Mostrar el gráfico
    plt.show()      
        


########################################### DERIVADAS

# Lista de dataframes
dataframes = [df_cont, df_diss, df_homo, df_ener, df_corr, df_ASM]
titles = ['Contraste', 'Dissimilarity', 'Homogeneidad', 'Energía', 'Correlación', 'ASM']

# Aplicar formato a cada dataframe
for i, df in enumerate(dataframes):
    # Calcula las derivadas en x e y
    df_dx = np.gradient(df.values, axis=1)
    df_dy = np.gradient(df.values, axis=0)
    # Calcula la derivada total como la raíz cuadrada de la suma de las derivadas al cuadrado
    df_total_deriv = np.sqrt(np.square(df_dx) + np.square(df_dy))
    # Crea un nuevo DataFrame con las derivadas totales
    df_total_deriv = pd.DataFrame(df_total_deriv, index=df.index, columns=df.columns)
    # Crear el heatmap
    plt.figure(figsize=(10, 8))
    #sns.heatmap(df_total_deriv, cmap="YlGnBu")
    #sns.heatmap(df_total_deriv, cmap="rocket")
    sns.heatmap(df_total_deriv, cmap="rocket")
    
    # Mostrar el heatmap
    plt.xlabel("Offset", fontsize=14)
    plt.ylabel("Ancho", fontsize=14)
    plt.title("Heatmap de la derivada de {}".format(titles[i]), fontsize=16)

    # Mostrar el heatmap
    plt.show()
    
    



# Dibujar superficie para cada dataframe
for i, df in enumerate(dataframes):
    # Calcula las derivadas en x e y
    df_dx = np.gradient(df.values, axis=1)
    df_dy = np.gradient(df.values, axis=0)
    # Calcula la derivada total como la raíz cuadrada de la suma de las derivadas al cuadrado
    df_total_deriv = np.sqrt(np.square(df_dx) + np.square(df_dy))
    # Crea un nuevo DataFrame con las derivadas totales
    df_total_deriv = pd.DataFrame(df_total_deriv, index=df.index, columns=df.columns)
    # Crear las coordenadas x, y y z
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(df.shape[1])
    y = np.arange(df.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = df_total_deriv.values

    # Dibujar la superficie
    ax.plot_surface(X, Y, Z, cmap='rocket')

    # Configurar los títulos de los ejes
    ax.set_xlabel('Offset')
    ax.set_ylabel('Ancho')
    ax.set_zlabel(titles[i])

    # Agregar título al gráfico
    ax.set_title("Superficie de la derivada de {}".format(titles[i]))

    # Mostrar el gráfico
    plt.show()    
    
    


########################################### DERIVADAS SEGUNDAS

# Lista de dataframes
dataframes = [df_cont, df_diss, df_homo, df_ener, df_corr, df_ASM]
titles = ['Contraste', 'Dissimilarity', 'Homogeneidad', 'Energía', 'Correlación', 'ASM']

# Aplicar formato a cada dataframe
for i, df in enumerate(dataframes):
    # Calcula las derivadas en x e y
    df_dx = np.gradient(df.values, axis=1)
    df_dy = np.gradient(df.values, axis=0)
    # Calcula la derivada total como la raíz cuadrada de la suma de las derivadas al cuadrado
    df_total_deriv = np.sqrt(np.square(df_dx) + np.square(df_dy))
    # Crea un nuevo DataFrame con las derivadas totales
    df_total_deriv = pd.DataFrame(df_total_deriv, index=df.index, columns=df.columns)
    # Calcula las derivadas SEGUNDAS en x e y
    df_dx = np.gradient(df_total_deriv.values, axis=1)
    df_dy = np.gradient(df_total_deriv.values, axis=0)
    # Calcula la derivada total como la raíz cuadrada de la suma de las derivadas al cuadrado
    df_total_deriv = np.sqrt(np.square(df_dx) + np.square(df_dy))
    # Crea un nuevo DataFrame con las derivadas totales
    df_total_deriv = pd.DataFrame(df_total_deriv, index=df.index, columns=df.columns)
    # Crear el heatmap
    plt.figure(figsize=(10, 8))
    #sns.heatmap(df_total_deriv, cmap="YlGnBu")
    #sns.heatmap(df_total_deriv, cmap="rocket")
    sns.heatmap(df_total_deriv, cmap="vlag")
    
    # Mostrar el heatmap
    plt.xlabel("Offset", fontsize=14)
    plt.ylabel("Ancho", fontsize=14)
    plt.title("Heatmap de la derivada de {}".format(titles[i]), fontsize=16)

    # Mostrar el heatmap
    plt.show()
    
    



# Dibujar superficie para cada dataframe
for i, df in enumerate(dataframes):
    # Calcula las derivadas en x e y
    df_dx = np.gradient(df.values, axis=1)
    df_dy = np.gradient(df.values, axis=0)
    # Calcula la derivada total como la raíz cuadrada de la suma de las derivadas al cuadrado
    df_total_deriv = np.sqrt(np.square(df_dx) + np.square(df_dy))
    # Crea un nuevo DataFrame con las derivadas totales
    df_total_deriv = pd.DataFrame(df_total_deriv, index=df.index, columns=df.columns)
    # Calcula las derivadas SEGUNDAS en x e y
    df_dx = np.gradient(df_total_deriv.values, axis=1)
    df_dy = np.gradient(df_total_deriv.values, axis=0)
    # Calcula la derivada total como la raíz cuadrada de la suma de las derivadas al cuadrado
    df_total_deriv = np.sqrt(np.square(df_dx) + np.square(df_dy))
    # Crea un nuevo DataFrame con las derivadas totales
    df_total_deriv = pd.DataFrame(df_total_deriv, index=df.index, columns=df.columns)
    # Crear las coordenadas x, y y z
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(df.shape[1])
    y = np.arange(df.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = df_total_deriv.values

    # Dibujar la superficie
    ax.plot_surface(X, Y, Z, cmap='vlag')

    # Configurar los títulos de los ejes
    ax.set_xlabel('Offset')
    ax.set_ylabel('Ancho')
    ax.set_zlabel(titles[i])

    # Agregar título al gráfico
    ax.set_title("Superficie de la derivada de {}".format(titles[i]))

    # Mostrar el gráfico
    plt.show()    
    
    
    


######################################################### Contraste vs. Homogeneidad

# Supongamos que tienes dos DataFrames llamados df1 y df2

# Obtén los valores de la columna "20" de cada DataFrame
values_dfcorr = df_corr[20].values
values_dfener = df_ener[20].values

# Obtén los índices de las filas
indices = df_cont.index

# Crea el gráfico 2D
fig = plt.figure(figsize=(10, 8))
plt.plot(indices, values_dfcorr, label="Contraste")
plt.scatter(indices, values_dfcorr)
plt.plot(indices, values_dfener, label="Homogeneidad")
plt.scatter(indices, values_dfener)

# Establece los valores del eje x
plt.xticks(indices)  # Puedes proporcionar aquí los valores específicos de los índices


# Agrega etiquetas y leyenda al gráfico
plt.xlabel("Ancho de ventana")
plt.ylabel("Valores para offset '20'")
plt.legend()

# Muestra el gráfico
plt.show()




###############################################


import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import feature
img = cv2.imread('2_isch.tif')
img2 = cv2.imread('2_nonisch.tif')


#obtiene dimensiones de img
#h, w, _ = img.shape
im=img2


H,W,_ = np.shape(im)
h = np.int0(H/2)
w = np.int0(W/2)
#h , w = 10000,13000
#define tamaño de la ventana
ventana = (h, w)
#cantidad de imágenes
x = 100
#posicion inicial de la ventana
pos_x = 0
pos_y = 0
#bucle para el ventaneo y generar las nuevas imagenes
for i in range(x):
    # Definir la posición final de la ventana
    fin_x = pos_x + w
    fin_y = pos_y + h
    
    # Seleccionar la sección de la imagen "im" que corresponde a la ventana
    ventana_im = im[pos_y:fin_y, pos_x:fin_x]
    
    # Guardar la imagen resultante con el nombre "im1", "im2", "im3", etc.
    nombre = 'Ventaneo_im2nonisch/im' + str(i+1) + '.tif'
    cv2.imwrite(nombre, ventana_im)
    
    # Actualizar la posición inicial de la ventana para la siguiente iteración
    pos_x += w
    
    # Si se alcanza el final de la imagen en el eje X, se reinicia en la posición inicial en X
    if pos_x + w > im.shape[1]:
        pos_x = 0
        pos_y += h
    
    # Si se alcanza el final de la imagen en el eje Y, se sale del bucle
    if pos_y + h > im.shape[0]:
        break




#--------------------------------------------

corr1 = -350
corr2 = 30
inifil = 3529 + corr1
finfil = 6626 + corr1
inicol = 100 + corr2
fincol = 5014 + corr2


cut1 = im[inifil:finfil,inicol:fincol,:]
plt.imshow(cut1)
# col aumenta hacia la derecha y (se supone) fila hacia abajo

cv2.imwrite('cx_isch1.tif', cut1)


#--------------------------------------------

corr1 = 0
corr2 = 100
inifil = 3961 + corr1
finfil = 7058 + corr1
inicol = 9536 + corr2
fincol = 14450 + corr2


cut1 = im[inifil:finfil,inicol:fincol,:]
plt.imshow(cut1)
# col aumenta hacia la derecha y (se supone) fila hacia abajo

cv2.imwrite('cx_isch2.tif', cut1)

#--------------------------------------------

corr1 = -600
corr2 = -400
inifil = 880 + corr1
finfil = 3977 + corr1
inicol = 3086 + corr2
fincol = 8000 + corr2


cut1 = im[inifil:finfil,inicol:fincol,:]
plt.imshow(cut1)
# col aumenta hacia la derecha y (se supone) fila hacia abajo

cv2.imwrite('cx_nonisch1.tif', cut1)

