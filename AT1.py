#Travail réalisé par MR.MEDJADJI ABDELDJALIL et MR.BENHAMEL OUSSAMA 
#Pour exécuter le processus de reconnaissance faciale, il faut tout d'abord exécuter le programme AT1.py. 
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import sys
import os
import math
print( "***********Travail réalisé par MR.MEDJADJI ABDELDJALIL et MR.BENHAMEL OUSSAMA  ***************")
#la partie de detection faciale pris depuis le site (https://www.datacorner.fr/reco-faciale-opencv/)
dirCascadeFiles = r"C:\Users\hp\Desktop\IV\M2\S1\ATelier\opencv-3.4\data\haarcascades_cuda"
cascadefile = dirCascadeFiles + "\haarcascade_frontalface_alt.xml"
classCascade = cv2.CascadeClassifier(cascadefile)

known_faces_filenames = [] # les noms des images dans la base des données

for (dirpath, dirnames, filenames) in os.walk('./assets'):
    known_faces_filenames.extend(filenames)
    break

#la fonction detecte le visage et le récupere dans crop_img ( si elle ne detecte aucun visage elle renvois un msg d'erreurs  )
def detection_fac (frm):
	gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
	B=False
	faces = classCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags = cv2.CASCADE_SCALE_IMAGE
	)
	for (x, y, w, h) in faces:
		cv2.rectangle(frm, (x, y), (x+w, y+h), (255,0, 0), 2)
	for (x, y, w, h) in faces:
		B=True
		crop_img = gray[y:y+h, x:x+w]
	if B:
		return(crop_img)
	else :
		return( "********************no face detected************************************ ")

#la fonction calcule le histogramme de chaque block 8*8 et les concataine
def DESC_LBP(cropp_img):
	resized = cv2.resize(cropp_img, (128,128), interpolation = cv2.INTER_AREA)
	discripteur=np.zeros(0).astype("i")
	rst=LBP(resized)
	for X in range ( 0 , resized.shape[0],8):
		for Y in range ( 0 , resized.shape[1],8):
			buffer =np.copy(rst[X:X+8,Y:Y+8])
			H=np.zeros(256).astype("i")
			for  a in buffer:
				for b  in a :
					H[int (b)]=H[int (b)]+1
			discripteur=np.append(discripteur, H)
	return ( discripteur)

#la fonction LBP
def LBP (A):
	A_pad=np.pad(A,(1, 1), 'symmetric' )
	LBP=np.zeros((A.shape[0],A.shape[1])).astype("f")
	for i in range(1, A_pad.shape[0]-1):
		for j in range (1, A_pad.shape[1]-1):
			kernel = np.copy(A_pad[i-1:i+2,j-1:j+2])
			chain=''
			l=np.copy(kernel[0])
			l=np.append(l, kernel[1,2])
			l1=np.copy(kernel[2])
			l1=l1[::-1]
			l=np.append( l,l1)
			l=np.append(l, kernel[1,0])
			l=l[::-1]
			for k in range( len ( l)):
				if l[k]<kernel[1,1]:
					l[k]=0
				else :
					l[k]=1
			for k in l :
				chain=chain+str(k)
			LBP[i-1,j-1] =int ( chain , 2)
	return (LBP)

#la fonction compare le discripteur avec tout les discripteurs de la database est elle renvois le nom de la valeurs minimale 
def MSE_LBP(dis):
	l=[] 
	val=1000 #initialisation de val et seuillage de mse
	for i in known_faces_filenames:
		imagee = cv2.imread('./assets/'+i)
		img=detection_fac(imagee)
		if isinstance(img,str):
			print( "**************NO FACE DETECTED IN ONE OF THE IMAGES*************")
			continue
		dcrp=DESC_LBP(img)
		summation=dis-dcrp
		summation=summation**2
		s=sum(summation)
		mse=s/len(dis)
		l.append(mse)
		if val>mse:
			val=mse
			pos=i	
	print( "les valeurs de la mésure MSE\n ",l)
	return(pos)
	
#la fonction calcule le histogramme de chaque block 8*8 et les concataine
def DESC_HOG (cropp_img):
	resized = cv2.resize(cropp_img, (128,128), interpolation = cv2.INTER_AREA)
	affiche=np.zeros(0).astype("i")
	discripteur=np.zeros(0).astype("i")
	for X in range ( 0 , resized.shape[0],8):
		for Y in range ( 0 , resized.shape[1],8):
			buffer =np.copy(resized[X:X+8,Y:Y+8])
			rst=HOG(buffer)
			H=rst.ravel()
			h=np.histogram(H, bins=[0,20,40,60,80,100,120,140,160,180])
			discripteur=np.append(discripteur, h[0])
	return ( discripteur)

#la fonction HOG 
def HOG (A):
	A_pad=np.pad(A,(1, 1), 'symmetric' )
	HOG=np.zeros((A.shape[0],A.shape[1])).astype("f")
	orientation =np.zeros((A.shape[0],A.shape[1])).astype("f")
	for i in range(1, A_pad.shape[0]-1):
		for j in range (1, A_pad.shape[1]-1):
			kernel = np.copy(A_pad[i-1:i+2,j-1:j+2])
			gx=kernel[1,2] - kernel[1,0]
			gy=kernel[2,1] - kernel[0,1]
			HOG[i-1,j-1] =math.sqrt(gx**2+gy**2)
			if ( gx!=0) and ( gy!=0):
				orientation[i-1,j-1]=np.arctan((gy/gx))
	orientation=np.degrees(orientation)
	return (orientation)


#la fonction compare le discripteur avec tout les discripteurs de la database est elle renvois le nom de la valeurs minimale 
def MSE_HOG(dis):
	val=1000 #initialisation de val et seuillage de mse 
	l=np.zeros(0)
	for i in known_faces_filenames:
		imagee = cv2.imread('./assets/'+i)
		img=detection_fac(imagee)
		if isinstance(img,str):
			print( "**************NO FACE DETECTED IN ONE OF THE IMAGES**************")
			continue
		dcrp=DESC_HOG(img)
		summation=dis-dcrp
		summation=summation**2
		s=sum(summation)
		mse=s/len(dis)
		l=np.append(l,mse)
		if val>mse:
				val=mse
				pos=i	
	l=l/100 #normalisation 
	print( "les valeurs de la mesure MSE \n",l)
	return(pos)

print( " *********Si Vous voulez faire une capture appuyer sur P et sur Q pour quitter **********")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
	print("*******Cannot open camera***********")
	exit()
name=""
name1=""
while True:
	ret, frame = cap.read() 
	if not ret:
		print("**************Can't receive frame (stream end?). Exiting *************")
		break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = classCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags = cv2.CASCADE_SCALE_IMAGE
	)
	
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0, 0), 2)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame, name, (x,y), font, 1.0, (255, 255, 255), 1) 
		cv2.putText(frame, name1, (x,y+h), font, 1.0, (255, 255, 255), 1) 
	cv2.imshow('frame', frame)
	if cv2.waitKey(1)==ord('p'):
		print('**********Entrer la methode souhaitée 1 pour LBP , 2 pour HOG **************')
		crop_img=detection_fac(frame)# traitement d'erreurs 
		if isinstance(crop_img,str):
			print( "************* NO FACE DETECTED BY THE CAMERA ********")
			break
		
		discripteur =DESC_LBP(crop_img)
		name =MSE_LBP(discripteur)
		name=name+"LBP"
		discripteur_hog=DESC_HOG(crop_img)
		name1=MSE_HOG(discripteur_hog)
		name1=name1+"HOG"
	if cv2.waitKey(1) == ord('q'):
		#pour quitter le programme 
		break
cap.release()
cv2.destroyAllWindows()
