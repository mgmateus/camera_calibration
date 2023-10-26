
#!/usr/bin/env python3
import cv2
import sys
import glob
import argparse
import datetime

import numpy as np
import matplotlib.pyplot as plt

from multipledispatch import dispatch 
from set_config import Config

CALIBRATION_PATH = '../calibration_files'
DATASET_PATH = '../dataset'

class Calib:

    @staticmethod
    def plot(img, dst):
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        ax1.axis('off')
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image', fontsize=30)
        ax2.axis('off')
        plt.show()

    @staticmethod
    def save_coefficients(mtx, dist, error, path):
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write("mtx", mtx)
        cv_file.write("dist", dist)
        cv_file.write("error", error)
        cv_file.release()
    
    
    def __init__(self, board_dim : list) -> None:
        self.__board_dim = (board_dim[0]-1, board_dim[1]-1)
        self.__criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
        self.__obj_points = []
        self.__img_points = []

        self.__coefficients_path = CALIBRATION_PATH
        self.__dataset_path = DATASET_PATH

        self.__obj_point = np.zeros((1, self.__board_dim[0]*self.__board_dim[1], 3), np.float32)
        self.__obj_point[0,:,:2] = np.mgrid[0:self.__board_dim[0], 0:self.__board_dim[1]].T.reshape(-1, 2)

        self.__calibrate_camera = {'ret' : None, 'mtx': None, "dist" : None , \
                                   'rvecs' : None, 'tvecs' : None, 'error' : None}
        
        self.__undistortion = ['default', 'remapping']
        
    @property
    def calibrate_camera(self):
        return self.__calibrate_camera

    def load_coefficients(self, path):
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

        # note we also have to specify the type to retrieve other wise we only get a
        # FileNode object back instead of a matrix
        camera_matrix = cv_file.getNode("mtx").mat()
        dist_matrix = cv_file.getNode("dist").mat()
        error = cv_file.getNode("error").mat()
        cv_file.release()

        self.__calibrate_camera['mtx'] = camera_matrix
        self.__calibrate_camera['dist'] = dist_matrix
        self.__calibrate_camera['error'] = error
        return [camera_matrix, dist_matrix]

    def calibrate(self, coefficients_path : str ="", coefficients_file_name : str = "", dataset_path : str = "", image_type : str = "", show : bool = True) -> bool:
        coefficients_full_path = (coefficients_path or self.__coefficients_path) +'/'+ \
                                 (coefficients_file_name or (str(datetime.datetime.now().strftime('%d-%m-%Y--%H:%M:%S'))+'.yaml'))
        
        dataset_full_path = (dataset_path or self.__dataset_path) +'/*.'+ image_type 
        images = glob.glob(dataset_full_path)
        for image in images:
            img_cv = cv2.imread(image)
            grayscale = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(grayscale, self.__board_dim,None)
            

            if ret:
                new_corners = cv2.cornerSubPix(grayscale, corners, (11,11),(-1,-1), self.__criteria)

                self.__obj_points.append(self.__obj_point)
                self.__img_points.append(new_corners)

                img_cv = cv2.drawChessboardCorners(img_cv, self.__board_dim, new_corners, ret)

            if show:
                cv2.imshow('image', img_cv)
                cv2.waitKey(1)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.__obj_points, self.__img_points, grayscale.shape[::-1], None, None)

        self.__calibrate_camera['ret'] = ret
        self.__calibrate_camera['mtx'] = mtx
        self.__calibrate_camera['dist'] = dist
        self.__calibrate_camera['rvecs'] = np.array(rvecs)
        self.__calibrate_camera['tvecs'] = np.array(tvecs)
        self.__calibrate_camera['error'] = self.__error()
        self.save_coefficients(mtx, dist, coefficients_full_path)
        cv2.destroyWindow('image')
        return True

    def undistortion(self, _type : str, image_path : str = "", save : bool = False):    
        if not _type in self.__undistortion:
            print("Undefined undistortion type!")
            sys.exit()

        mtx = self.__calibrate_camera['mtx']
        dist = self.__calibrate_camera['dist']
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        x, y, w, h = roi
        if _type == 'default':
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            
        else:
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        dst = dst[y:y+h, x:x+w]      
        if save:
            name = image_path.split("/")[3]
            _type = name.split('.')[1]
            image_path = image_path  - name
            cv2.imwrite(image_path+'output/'+name+'_undistorted.'+_type)
        self.plot(img, dst)
        

    def __error(self) -> float:
        mean_error = 0
        for i in range(len(self.__obj_points)):
            imgpoints2, _ = cv2.projectPoints(self.__obj_points[i], self.__calibrate_camera['rvecs'][i], self.__calibrate_camera['tvecs'][i], \
                                               self.__calibrate_camera['mtx'], self.__calibrate_camera['dist'])
            error = cv2.norm(self.__img_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        return mean_error/len(self.__obj_points)
    
    def __str__(self) -> str:
        string = f"Camera calibrated: {self.__calibrate_camera['ret']} \n \
                  \n Camera Matrix: {self.__calibrate_camera['mtx']} \n \
                  \n Distortion Parameters: {self.__calibrate_camera['dist']} \n \
                  \n Rotation Vectors: {self.__calibrate_camera['rvecs']} \n \
                  \n Translation Vectors: {self.__calibrate_camera['tvecs']} \n\
                  \n Error: {self.__calibrate_camera['error']} \n\
                  " 
        return string


class Calibration:
    @staticmethod
    def plot(img, dst):
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        ax1.axis('off')
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image', fontsize=30)
        ax2.axis('off')
        plt.show()
        return True

    @staticmethod
    def save_coefficients(mtx, dist, err, path):
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write("mtx", mtx)
        cv_file.write("dist", dist)
        cv_file.write("err", err)
        cv_file.release()
        return True
    
    @staticmethod
    def load_coefficients(path):
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        camera_matrix = cv_file.getNode("mtx").mat()
        dist_matrix = cv_file.getNode("dist").mat()
        err = cv_file.getNode("err")
        cv_file.release()

        return camera_matrix, dist_matrix, err

    def __init__(self, dataset_info : dict) -> None:
        self.__content = dataset_info['content']
        self.__criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
        self.__obj_point = np.zeros((1, self.__content['board_dim'][0]*self.__content['board_dim'][1], 3), np.float32)
        self.__obj_point[0,:,:2] = np.mgrid[0:self.__content['board_dim'][0], 0:self.__content['board_dim'][1]].T.reshape(-1, 2)
        self.__obj_points = []
        self.__img_points = []

        self.__calibrate_camera = {'ret' : None, 'mtx': None, "dist" : None , \
                                   'rvecs' : None, 'tvecs' : None, 'error' : None}
        
    def __str__(self) -> str:
        string = f"\nCamera calibrated: {self.__calibrate_camera['ret']} \n \
                  \n Camera Matrix: {self.__calibrate_camera['mtx']} \n \
                  \n Distortion Parameters: {self.__calibrate_camera['dist']} \n \
                  \n Error: {self.__calibrate_camera['err']} \n\
                  \n Rotation Vectors: {self.__calibrate_camera['rvecs']} \n \
                  \n Translation Vectors: {self.__calibrate_camera['tvecs']} \n\
                  " 
        return string
        
    def _error(self, mtx, dist, rvecs, tvecs) -> float:
        mean_error = 0
        for i in range(len(self.__obj_points)):
            imgpoints2, _ = cv2.projectPoints(self.__obj_points[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(self.__img_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        return mean_error/len(self.__obj_points)

    def calibrate(self, calibration_file : str = '', show : bool = False):
        dataset_full_path = self.__content['dataset_path'] +'/*.'+ self.__content['image_type'] 
        calibration_full_path = self.__content['calibration_path'] +'/'+ \
                                (calibration_file or (str(datetime.datetime.now().strftime('%d-%m-%Y--%H:%M:%S'))+'.yaml'))
        images = glob.glob(dataset_full_path)
        for image in images:
            img_cv = cv2.imread(image)
            grayscale = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(grayscale, self.__content['board_dim'],None)
            if ret:
                new_corners = cv2.cornerSubPix(grayscale, corners, (11,11),(-1,-1), self.__criteria)

                self.__obj_points.append(self.__obj_point)
                self.__img_points.append(new_corners)

                img_cv = cv2.drawChessboardCorners(img_cv, self.__content['board_dim'], new_corners, ret)

            if show:
                cv2.imshow('image', img_cv)
                cv2.waitKey(1)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.__obj_points, self.__img_points, grayscale.shape[::-1], None, None)
        err = self._error(mtx, dist, rvecs, tvecs)
        self.__calibrate_camera['ret'] = ret
        self.__calibrate_camera['mtx'] = mtx
        self.__calibrate_camera['dist'] = dist
        self.__calibrate_camera['rvecs'] = np.array(rvecs)
        self.__calibrate_camera['tvecs'] = np.array(tvecs)
        self.__calibrate_camera['err'] = err
        self.save_coefficients(mtx, dist, err, calibration_full_path)
        cv2.destroyWindow('image')
        return True
    
        
    def undistort(self, _type : str, image : str, calibration_file : str = "", save : bool = True):
        if type(self.__calibrate_camera['mtx']) == np.ndarray:
            mtx, dist = self.__calibrate_camera['mtx'], self.__calibrate_camera['dist']
        else:
            mtx, dist, _ = self.load_coefficients('../calibration_files/'+calibration_file)
        
        img = cv2.imread(image)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        x, y, w, h = roi
        if _type == 'default':
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            
        else:
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        dst = dst[y:y+h, x:x+w]      
        if save:
            full_name_image = image.split("/")[-1]
            print(full_name_image)
            name, _type = full_name_image.split('.')
            image = image.replace(full_name_image, "")
            cv2.imwrite(image+'output/'+name+'_undistorted.'+_type, img)
        self.plot(img, dst)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Configuration')
    
    parser.add_argument('--calibrate', type= str, required=False, help='Generate a calibration file.')
    parser.add_argument('--dataset', type= str, required=False, help='A valid dataset name e.g.(chesseboard_aruco, chesseboard_kaggle)')
    parser.add_argument('--undistortion', type= str, required=False, help='A valid dataset name e.g.(chesseboard_aruco, chesseboard_kaggle)')
    parser.add_argument('--type', type= str, required=False, help='Set a undistortion type: default or remapping')
    parser.add_argument('--image_to_undistort', type= str, required=False, help='/home/path/to/image_file.*')
    parser.add_argument('--calibration_file', type= str, required=False, help='File name. e.g. (your_file.yaml) default name is d-%m-%Y--%H:%M:%S.yaml')
    
    args = parser.parse_args()
    calibrate = args.calibrate
    dataset = args.dataset
    undistortion = args.undistortion
    _type = args.type
    image_to_undistort = args.image_to_undistort
    calibration_file = args.calibration_file

    if calibrate == 'True' and dataset:
        ret, dataset_info = Config.load_dataset_info(dataset= dataset)
        if ret:
            calibration = Calibration(dataset_info= dataset_info)
            calibration.calibrate(calibration_file= calibration_file, show= True)
            print(calibration)
            if undistortion == 'True':
                if _type and image_to_undistort:
                    calibration.undistort(_type= _type, image= image_to_undistort, calibration_file= calibration_file)
                else:
                    print('Define a undistortion type and a image to undistort.')
                    sys.exit()

        else:
            print('Invalid dataset name!')
            sys.exit()
    
    elif calibrate == 'true' and not dataset:
        print('Define a dataset to calibrate.')
        sys.exit()
    
    elif undistortion == 'True' and dataset:
        if _type and image_to_undistort and calibration_file:
            ret, dataset_info = Config.load_dataset_info(dataset= dataset)
            if ret:
                calibration = Calibration(dataset_info= dataset_info)
                calibration.undistort(_type= _type, image= image_to_undistort, calibration_file= calibration_file)
        else:
            print('Define a dataset, a undistortion type, a image and calibration file to undistort.')
            sys.exit()

    

   
