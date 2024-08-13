# Importação
import PIL
from skimage.feature import peak_local_max
from skimage.segmentation import  watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2

endereco_imagem = './imagens/seed-1.jpg'

# Abrir imagem
image = cv2.imread(endereco_imagem)
shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

#Cria uma janela
cv2.namedWindow('Imagem', cv2.WINDOW_NORMAL)

# Definir o tamanho da janela
cv2.resizeWindow('Imagem', 800, 600)  # largura=800, altura=600

# Exibir a imagem
cv2.imshow('Imagem', image)

# Converte para escala de cinza e aplica um threshold
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#Cria uma janela
cv2.namedWindow('Thresh', cv2.WINDOW_NORMAL)

# Definir o tamanho da janela
cv2.resizeWindow('Thresh', 800, 600)  # largura=800, altura=600
cv2.imshow("Thresh", thresh)

# Calcula a distância euclidiana entre cada pixel e o pixel 0
D = ndimage.distance_transform_edt(thresh)

#Utilizado para achar os picos locais (Encontrar as bordas dos objetos na imagem)
localMax = peak_local_max(D, indices=False, min_distance=20,labels=thresh)


# A função label gera um array onde os objetos na entrada são rotulados com um índice inteiro.
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]

# Executa o algoritmo para encontrar o número de elementos
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} - elementos encontrados".format(len(np.unique(labels)) - 1))

# Para cada elemento encontrado
for label in np.unique(labels):
	
    # Ele identifica a label de brackground como 0
	if label == 0:
		continue

	# Cria uma máscara de 0
	mask = np.zeros(gray.shape, dtype="uint8")
	mask[labels == label] = 255
	
    # Detecta os contornos dos objetos pelo 0
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	
    # Desenha um círculo no objeto encontrado
	((x, y), r) = cv2.minEnclosingCircle(c)
	cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
	cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Output', 800, 600)  # largura=800, altura=600

cv2.imshow("Output", image)
cv2.waitKey(0)