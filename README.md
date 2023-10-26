# Running
clone the repository.
```
git clone git@github.com:mgmateus/camera_calibration.git
cd camera_calibration
cd scripts
```
## Calibration with an exisistent dataset
```
python3 calibration.py --calibrate True --calibration_file kaggle.yaml --dataset chessboard_kaggle
```
## Calibration and undistortion with an exisistent dataset
```
python3 calibration.py --calibrate True --calibration_file kaggle.yaml --dataset chessboard_kaggle --undistortion True --type deafault --image_to_undistort define_a_image.png 
```
## Using your's dataset
Run this module to generate a config file to your dataset. After this, just need run a calibration or calibration and undistortion to create a calibration file. With a calibration file you can run just de undistortion on a image of this dataset.
```
python3 set_config.py --dataset_path /path/to/your/dataset --width number_of_squares_on_width_in_chessboard --height number_of_squares_on_height_in_chessboard --image_type type_of_your_images --calibration_path /path/to/your/calibration/file
```
