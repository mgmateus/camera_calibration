import os
import sys
import yaml
import argparse


class Config:
    @staticmethod
    def create_new_config(dataset_path : str, \
                 board_dim : list, \
                 image_type : str, \
                 calibration_path : str) -> bool:
        """
        Create a new dataset configuration to camera calibrate.

        Input:

        dataset_path: Full path to the image directory.
        board_dim: Board's dimension.
        image_type: Extension of image files: .e.g.(png, jpeg)
        calibration_path: Full path to the calibration files directory.

        Output:

        :return True
        """
        
        dataset_name = dataset_path.split('/')[-1]
        board_dim = [board_dim[0]-1, board_dim[1]-1]
        args = [{
            'dataset' : dataset_name, 
            'content' : {
                'dataset_path' : dataset_path,
                'board_dim' : board_dim, 
                'image_type' : image_type, 
               'calibration_path' : calibration_path
            }
        }]
        try:
            with open('../config/config.yaml', 'r') as f:
                args_yaml = yaml.safe_load(f)

            if args_yaml and not args[0] in args_yaml: 
                args_yaml += args
                with open('../config/config.yaml', 'w') as f:
                    yaml.dump(args_yaml,f,sort_keys=False)
                print("Dataset was added with success!")
            elif not args_yaml:
                with open('../config/config.yaml', 'w') as f:
                    yaml.dump(args,f,sort_keys=False)
                print("Dataset was created with success!")
            else:
                print("Dataset already exists!")
                pass
        except FileNotFoundError:
            with open('../config/config.yaml', 'w') as f:
                yaml.dump(args,f,sort_keys=False)

            print("Config file created with success!")
            
        return True
    
    @staticmethod
    def load_dataset_info(dataset : str) -> list:
        """
        Load the dataset information.

        Input:

        dataset: Dataset's name.

        Output:

        return True, arg -- If dataset exists.
        return False -- If dataset not exist.
        """
        with open('../config/config.yaml', 'r') as f:
            args_yaml = yaml.safe_load(f)

        for arg in args_yaml:
            if arg['dataset'] == dataset:
                return True, arg
            
        return False, None
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration')
    
    parser.add_argument('--dataset_path', type= str, required=True, help='/home/path/to/your/dataset')
    parser.add_argument('--width', type= int, required=True, help="Number of squares on width in chessboard")
    parser.add_argument('--height', type= int, required=True, help="Number of squares on high in chessboard")
    parser.add_argument('--image_type', type= str, required=True, help='e.g. (png, jpg, jpeg)')
    parser.add_argument('--calibration_path', type= str, required=True, help='/home/path/to/your/calibration_files')

    args = parser.parse_args()

    board_dim = [args.width, args.height]
    Config.create_new_config(dataset_path= args.dataset_path, board_dim= board_dim, \
                             image_type= args.image_type, calibration_path= args.calibration_path)