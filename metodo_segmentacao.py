# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:51:50 2016

@author: lagan11
"""
#Import bibliotecas
import gdal
import numpy as np
from skimage import measure
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import matplotlib.pyplot as plt
nb_classes = 2

def load_dataset(file_amostras_classes,file_img):

        gdal.AllRegister()
        #ler imagem
        ds_img = gdal.Open(file_img, gdal.GA_ReadOnly)
        #Transformar as amostras em Labels regioes
        ##ler amostras como array
        
        ds = gdal.Open(file_amostras_classes, gdal.GA_ReadOnly)
        amostras_array = ds.GetRasterBand(1).ReadAsArray()
        #obter valores classe_id do array
        #classes_amostras=np.unique(amostras_array)
        #excluir nodata

        #obter amostras divididas
        amostras_labels = measure.label(amostras_array)
        #Obter as propriedades dos labels separados
        amostras_prop = measure.regionprops(amostras_labels)
        #Lado do quadrado
        
        compr = 83#min(amostras_prop[0].image.shape[0],amostras_prop[0].image.shape[0])
        print 'comprimento e largura array: ', compr
        #percorrer as regioes e obter as classes e amostras de treinamento
        classes = np.array(range(len(amostras_prop)))
        amostras_treinamento = np.zeros((len(amostras_prop),3,compr,compr),dtype=np.int16)
        
        for i, reg in enumerate(amostras_prop):
            print 'i: ',i
            print 'reg.image.shape[0]: ',reg.image.shape
            print 'Classe: ',np.unique(amostras_array[reg.coords[:,0],reg.coords[:,1]])
            
            #Obter as classes  por meio das coordenadas das regioes
            classes[i]=np.unique(amostras_array[reg.coords[:,0],reg.coords[:,1]])[0]
            #Obter dados da imagem por meio das coordenadas das regioes
            amostras_treinamento[i,0,:,:] = ds_img.GetRasterBand(1).ReadAsArray()[reg.coords[:,0],reg.coords[:,1]].reshape(reg.image.shape[0],reg.image.shape[1])[:compr,:compr]
            amostras_treinamento[i,1,:,:] = ds_img.GetRasterBand(2).ReadAsArray()[reg.coords[:,0],reg.coords[:,1]].reshape(reg.image.shape[0],reg.image.shape[1])[:compr,:compr]
            amostras_treinamento[i,2,:,:] = ds_img.GetRasterBand(3).ReadAsArray()[reg.coords[:,0],reg.coords[:,1]].reshape(reg.image.shape[0],reg.image.shape[1])[:compr,:compr]
        # convert class vectors to binary class matrices
        print 'classes.shape: ',classes.reshape(len(amostras_prop),1).shape
        print 'unique(classes): ',np.unique(classes)
        Y_train = np_utils.to_categorical(classes.reshape(len(amostras_prop),1), nb_classes)
            
        X_train = amostras_treinamento.astype('float32')
        X_train /=255
             
        return X_train, Y_train
def rolling2d(a,win_h,win_w,step_h,step_w):

    h,w = a.shape
    #shape = ( a.shape[0]*a.shape[1] , win_h , win_w)
    shape= ((h-win_h)/step_h + 1)  * ((w-win_w)/step_w + 1) , win_h , win_w
    print 'shape: ',shape
    strides = (step_w*a.itemsize, h*a.itemsize,a.itemsize)
       
    print 'strides: ',strides
    a= np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return a.copy()    
#Reconhecimento com rede neural convolutiva
def make_network(img_channels, img_rows, img_cols):
   model = Sequential()
 
   model.add(Convolution2D(15, 3, 3, border_mode='same',input_shape=(img_channels, img_rows, img_cols)))
   model.add(Activation('relu'))
   model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th")) 
   model.add(Convolution2D(15, 3, 3))
   model.add(Activation('relu'))
   model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th")) 
   #model.add(Convolution2D(64, 3, 3, border_mode='same'))
   #model.add(Activation('relu'))
   #model.add(Convolution2D(64, 3, 3))
   #model.add(Activation('relu'))
   #model.add(MaxPooling2D(pool_size=(2, 2)))
   #model.add(Dropout(0.25))
 
   model.add(Flatten())
   model.add(Dense(512))
   model.add(Activation('relu'))
   #model.add(Dropout(0.5))
   model.add(Dense(nb_classes))
   model.add(Activation('softmax'))
 
   return model
   
def train_model(model, X_train, Y_train, X_test, Y_test,nb_epoch, batch_size):
 
   sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
   model.compile(loss='categorical_crossentropy', optimizer=sgd)
 
   model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size,
             validation_split=0.35, show_accuracy=True, verbose=1)
 
   #print('Testing...')
   #return model.evaluate(X_test, Y_test,
   #                    batch_size=batch_size, verbose=0, show_accuracy=True)
   #print('Test accuracy: {0}'.format(res[1]))
#dados
#amostras em raster
#file_amostras= '/media/ruiz/Documentos/Pesquisas/Metodo_Segmentacao/Rasters/amostras_quad_alvos.tif'
file_amostras_img='/home/ruiz/Documentos/Pesquisas/RN_conv/Rasters/amostras_A.tif'
#arquivo ortoimagem
file_orto= '/home/ruiz/Documentos/Pesquisas/Kmeans_Segmentacao/orto_img/orto.tif'
file_amostras_classes = '/home/ruiz/Documentos/Pesquisas/RN_conv/Rasters/amostras_inteiro_A.tif'
#carregar os dados
X_train, Y_train = load_dataset(file_amostras_classes,file_amostras_img)
X_test, Y_test = X_train, Y_train
#criar modelo
modelo=make_network(3,X_train.shape[2],X_train.shape[2])
#treinar o modelo
train_model(modelo,X_train, Y_train,X_test, Y_test,1000,10)
#Verificar na imagem
##ler amostras como array
ds_img_orto= gdal.Open(file_orto, gdal.GA_ReadOnly)
amostras_array = ds_img_orto.GetRasterBand(1).ReadAsArray(0,0,200,200)
#criar blocos com strides
blocos=rolling2d(amostras_array,83,83,3,3)
#criar array
array_aux=np.zeros((blocos.shape[0],3,blocos.shape[1],blocos.shape[2]),dtype=np.int16)
#Inserir blocos banda 1
array_aux[:,0,:,:]=blocos
#criar blocos banda 2
amostras_array = ds_img_orto.GetRasterBand(2).ReadAsArray(0,0,200,200)
blocos=rolling2d(amostras_array,83,83,3,3)
#Inserir blocos banda 1
array_aux[:,1,:,:]=blocos
#criar blocos banda 2
amostras_array = ds_img_orto.GetRasterBand(3).ReadAsArray(0,0,200,200)
blocos=rolling2d(amostras_array,83,83,3,3)
#Inserir blocos banda 1
array_aux[:,2,:,:]=blocos
array_aux/=255
# predicao
#class_pred= modelo.predict_classes(array_aux)