import numpy as np
import cv2 as cv
import math

img = cv.imread('../dataset/chessboard_kaggle/Im_R_14.png')

# grab the dimensions of the image
(h, w, _) = img.shape

# set up the x and y maps as float32
map_x = np.zeros((h, w), np.float32)
map_y = np.zeros((h, w), np.float32)

scale_x = 1
scale_y = 1
center_x = w/2
center_y = h/2
radius = w/2
#amount = -0.75   # negative values produce pincushion
amount = 0.75   # positive values produce barrel

# create map with the barrel pincushion distortion formula
for y in range(h):
    delta_y = scale_y * (y - center_y)
    for x in range(w):
        # determine if pixel is within an ellipse
        delta_x = scale_x * (x - center_x)
        distance = delta_x * delta_x + delta_y * delta_y
        if distance >= (radius * radius):
            map_x[y, x] = x
            map_y[y, x] = y
        else:
            factor = 1.0
            if distance > 0.0:
                factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
            map_x[y, x] = factor * delta_x / scale_x + center_x
            map_y[y, x] = factor * delta_y / scale_y + center_y
            

# do the remap
dst = cv.remap(img, map_x, map_y, cv.INTER_LINEAR)

# save the result
#cv.imwrite('lena_pincushion.jpg',dst)
cv.imwrite('../dataset/chessboard_kaggle/Im_R_14_barrel.png',dst)

# show the result
cv.imshow('src', img)
cv.imshow('dst', dst)

cv.waitKey(0)
cv.destroyAllWindows()



string = f"\n \
    Created dataset : chessboard_aruco (11 X 8) \n \
    ---------------- files : 21 \n \
    ---------------- Image Type : jpeg \n \n \
    Comparative dataset : chessboard_kaggle (12 X 8) \n \
    ---------------- files : 40 \n \
    ---------------- Image Type : png \n \
    "
    print(string)

    parser = argparse.ArgumentParser(description='Camera calibration')
    
    parser.add_argument('-c', type=bool, required=False)
    parser.add_argument('--width', type=int, required=False)
    parser.add_argument('--height', type=int, required=False)
    parser.add_argument('--dataset_path', type=str, required=False)
    parser.add_argument('--image_type', type=str, required=False)
    parser.add_argument('--coeffs_path', type=str, required=False)
    parser.add_argument('--coeffs_file', type=str, required=False)
    parser.add_argument('--coeffs_file_full_path', type=str, required=False)
    parser.add_argument('--image_path', type=str, required=False)
    parser.add_argument('-u', type=str, required=False)


    args = parser.parse_args()

    if args.width is None or args.height is None:
        print("Board's dimension can't be None args!")
        sys.exit()

    calib = Calibration(board_dim=(args.width, args.height))
    if args.c:

        if not args.image_type:
            print("Define a image type!")
            sys.exit()
        
        if len(glob.glob((args.dataset_path or DATASET_PATH) +"/*."+args.image_type)) == 0:
            print("Image number in dataset need's bigger than 0!")
            sys.exit()

        calib.calibrate(args.coeffs_path, args.coeffs_file, args.dataset_path, args.image_type)

    if args.u:
        if not args.image_path:
            print("Image path can't be None!")
            sys.exit()

        calib.undistorted(args.image_path)
    
    
    #calib = Calibration((11,8))
    #calib.calibrate(coefficients_file_name="teste.yaml", dataset_path="../dataset/chessboard_aruco", image_type="jpeg")

    #calib.undistortion('default', '../dataset/chessboard_aruco/WhatsApp Image 2023-10-06 at 22.49.03 (1).jpeg')


    #calib = Calibration((12,8))
    #calib.calibrate(dataset_path="../dataset/chessboard_kaggle", image_type="png")
    #print(calib)
    #calib.undistortion('remapping', '../dataset/chessboard_kaggle/Im_R_12.png')

        


