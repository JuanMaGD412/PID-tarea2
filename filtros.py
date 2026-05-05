import cv2
import numpy as np

# a. Filtro de media
def filtro_media(img):
    return cv2.blur(img, (3, 3))


# b. Filtro de mediana
def filtro_mediana(img):
    return cv2.medianBlur(img, 3)


# c. Filtro logaritmico
def filtro_logaritmico(img):
    img_float = np.float32(img)
    c = 255 / np.log(1 + np.max(img_float))
    log = c * np.log(1 + img_float)
    return np.uint8(log)


# d. Filtro de cuadro normalizado
def filtro_normalizado(img):
    kernel = np.ones((3, 3), np.float32) / 9
    return cv2.filter2D(img, -1, kernel)


# e. Filtro gaussiano
def filtro_gaussiano(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

    # f. Filtro Laplace
def filtro_laplace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)


# g. Filtro Sobel
def filtro_sobel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(sobel)

    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)


# h. Filtro Canny
def filtro_canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
