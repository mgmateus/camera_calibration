import cv2
import numpy as np
import glob
# Parâmetros do tabuleiro de xadrez
chessboard_size = (11, 7)  # Número de cantos internos no tabuleiro (largura x altura)
# Preparar pontos do objeto 3D
objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
# Listas para armazenar pontos do objeto 3D e pontos da imagem 2D
object_points = []
image_points = []
list_of_image_files = glob.glob('../dataset/chessboard_kaggle/*.png')
# Carregar e processar cada imagem
for image_file in list_of_image_files:
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detectar cantos do tabuleiro de xadrez
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, 
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                    cv2.CALIB_CB_FAST_CHECK + 
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE)
    # Se os cantos forem encontrados, adicione os pontos do objeto e da imagem
    if ret:
        object_points.append(objp)
        image_points.append(corners)
        # Desenhar e exibir os cantos
        cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
        cv2.imshow('img', image)
        cv2.waitKey(1)
cv2.destroyAllWindows()
# Calibrar a câmera
ret, k, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, gray.shape[::-1], None, None
)
print("Matriz de calibração k:\n", k)
print("Distorção:", dist.ravel())

# Carregar uma imagem de teste
test_image = cv2.imread('../dataset/chessboard_kaggle/Im_R_10.png')
# Corrigir a distorção da imagem
undistorted_image = cv2.undistort(test_image, k, dist, None, k)
# Exibir a imagem original e a imagem corrigida lado a lado
combined_image = np.hstack((test_image, undistorted_image))
cv2.imshow('Original vs Undistorted', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()