# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:51:50 2016

@author: lagan11
"""
#Import bibliotecas
import gdal
import numpy as np
from skimage import measure
from sknn.mlp import Classifier, Convolution, Layer
#arquivo ortoimagem
#Testar essa modificacao
a=np.array([1,2,3,4,5,6])
file_img= '/media/lagan11/Sistema/LuisFernando/Pesquisas/OBIA_C50/Rasters/orto/orto.tif'
#Ler ortoimagem
gdal.AllRegister()
ds_img=gdal.Open(file_img,gdal.GA_ReadOnly)
#amostras em raster
file_amostras= '/media/lagan11/Sistema/LuisFernando/Pesquisas/Metodo_segmentacao/Rasters/amostras_quad_alvos.tif'
#Transformar as amostras em Labels regioes
##ler amostras como array
ds = gdal.Open(file_amostras, gdal.GA_ReadOnly)
amostras_array = ds.GetRasterBand(1).ReadAsArray()
#obter valores classe_id do array
#classes_amostras=np.unique(amostras_array)
#excluir nodata
#classes_amostras=np.delete(classes_amostras,np.where(np.in1d(classes_amostras,ds.GetRasterBand(1).GetNoDataValue()))[0][0])

#obter amostras divididas
amostras_labels = measure.label(amostras_array)
#Obter as propriedades dos labels separados
amostras_prop = measure.regionprops(amostras_labels)
#percorrer as regioes e obter as classes e amostras de treinamento
classes = np.array(range(len(amostras_prop)))
amostras_treinamento = np.zeros((99*99,35))
for i, reg in enumerate(amostras_prop):
    print 'i: ',i
    #Obter as classes  por meio das coordenadas das regioes
    classes[i]=np.unique(amostras_array[reg.coords[:,0],reg.coords[:,1]])[0]
    #Obter dados da imagem por meio das coordenadas das regioes
    amostras_treinamento[:,i] = ds_img.GetRasterBand(1).ReadAsArray()[reg.coords[:,0],reg.coords[:,1]][:99*99]
    #amostras_treinamento[:,1,i] = ds_img.GetRasterBand(2).ReadAsArray()[reg.coords[:,0],reg.coords[:,1]][:99*99]
    #amostras_treinamento[:,2,i] = ds_img.GetRasterBand(3).ReadAsArray()[reg.coords[:,0],reg.coords[:,1]][:99*99]
    
#Reconhecimento com rede neural convolutiva
nn = Classifier(layers=[Convolution("Rectifier", channels=8, kernel_shape=(3,3)), Layer("Softmax")],learning_rate=0.02, n_iter=10)
nn.fit(amostras_treinamento,classes)

