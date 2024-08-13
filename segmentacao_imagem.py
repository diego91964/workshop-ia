import os  #Utilitários de sistema operacional
import PIL #Operações com imagens
from PIL import Image, ImageOps # Utilitário para imagem
from skimage.segmentation import slic
from skimage.util import img_as_float
import numpy as np
from skimage.io import imsave, imread


folder_dst = './segmentado'
endereco_imagem = './imagens/foliolo-1.jpg'
nome_segmento = 'seg'

 # Inicia a segmentação para remoção do background
image = Image.open(endereco_imagem)
img = img_as_float(image)

# Executa segmentação utilizando o método superpixel
segments_slic = slic(img, n_segments=2, compactness=10, sigma=1, start_label=1)

# Para cada segmento encontrado (Fundo e folíolo)
for index in np.unique(segments_slic):
        
    # Cria uma pasta para armazenar os segmentos
    folder_to_segment = folder_dst + "/segment_{}".format(str(index))
    
    if not os.path.isdir(folder_to_segment):
        os.mkdir(folder_to_segment)

    # Abre a imagem original utilizando scikit-image
    skimage = imread(endereco_imagem)

    # Altera os pixels do segmento para branco
    skimage[segments_slic== index] = [255,255,255]

    # Cria nome da nova imagem
    #segmented_image_name = folder_to_segment.replace('.jpg','')
    segmented_image_name = "{}/{}segmented_{}.jpg".format(folder_to_segment, nome_segmento, index)
    print("Salvando em "+ segmented_image_name)
    imsave (segmented_image_name, skimage)
    