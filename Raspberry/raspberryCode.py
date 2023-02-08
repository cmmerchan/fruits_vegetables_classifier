import cv2
import numpy as np
import tensorflow as tf

labels = ['freshoranges', 'rottenapples', 'rottenoranges', 'freshapples', 'freshpeppers', 'rottenpeppers']
model = tf.keras.models.load_model('modelV1.h5') #Se carga el modelo
frame = cv2.imread("orangeTest.jpeg") #Se añade la imagen a realizar la predicción

#Se ajusta la imagen y se  la recorta para obtener el enfoque deseado
frame= cv2.resize(frame,dsize=(350,240), interpolation = cv2.INTER_CUBIC)
target = frame[20:200,50:300] #Recorta la imagen
target= cv2.resize(target,dsize=(240,240), interpolation = cv2.INTER_CUBIC) #Size use for training model

#Para realizar las inferencias es necesario convertir la imagen a vector
np_image_data = np.asarray(target)
img_tensor = np.expand_dims(np_image_data, axis=0) 
img_tensor = img_tensor/255
pred = model.predict(img_tensor)

#Se toma la siguiente decisión en caso que el modelo no clasifique la imagen, en este caso fruta o verdura con un theshold
#establecido se dice que su madurez está en un nivel intermedio y se lo clasifica como consumo animal
threshold = 0.5

if(pred.max()>=threshold):
  print("Precisión: "+ str(pred[::,pred.argmax()]*100))
  if(labels[pred.argmax()][:6]=="rotten"):
    texto ="MERMA"
  else:
    texto ="CONSUMO HUMANO"
else:
  texto ="COSUMO ANIMAL"

#Para colocar texto en una imagen final
#Características del texto
ubicacion = (50,210)
font = cv2.FONT_HERSHEY_PLAIN
tamañoLetra = 1
colorLetra = (0,0,0)
grosorLetra = 1

#Escribir texto
cv2.putText(target, texto, (50,210), font, tamañoLetra, colorLetra, grosorLetra)
cv2.imshow('imagen',target)
cv2.waitKey(30000)

