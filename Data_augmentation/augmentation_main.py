def main():
    import os
    import cv2
    import unittest

    # Importing the main pileine function
    from Supporting_code_aug.augutilities import create_dataset_from_images

    print('Initiating data augmentation pipline')

    # Specifying the constants
    # Paths to input/output folders
    ROUND_MARKER_FOLDER = './Data_augmentation/Example_markers/Round'
    SQUARE_MARKER_FOLDER = './Data_augmentation/Example_markers/Square'

    TRAINING_RAW_DATA_PATH = './Data_augmentation/Sample_data/sample_images1024_training'
    TRAINING_OUTPUT_PATH = './Data_augmentation/Sample_data/For_model_development/datasets/training_data'

    HOLDOUT_RAW_DATA_PATH = './Data_augmentation/Sample_data/sample_images1024_holdout'
    HOLDOUT_OUTPUT_PATH = './Data_augmentation/Sample_data/For_model_development/datasets/holdout_data'

    # N of images to prapare for model development (usually > 1000)
    N_TRAINING_IMAGES = 50
    N_HOLDOUT_IMAGES = 25

    # Pre-processing variables
    BLACK_AND_WHITE = False
    EDGE_TRANSFORM = False
    RANDOM_BRIGHTNESS_SHIFT = True
    RANDOM_IMAGE_ROTATION = True
    RANDOM_MARKER_ROTATION = True


    # Performing unit tests on the pipelines functions

    # Load the test suite from the test_image_processing module
    print('Performing unit testing')
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='./Data_augmentation/Supporting_code_aug', pattern='unittests.py')

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)


    # Loading markers
    print('Loading markers')
    round_marker_files = [os.path.join(ROUND_MARKER_FOLDER, file) for file in os.listdir(ROUND_MARKER_FOLDER) if file.lower().endswith('.png')]
    square_marker_files = [os.path.join(SQUARE_MARKER_FOLDER, file) for file in os.listdir(SQUARE_MARKER_FOLDER) if file.lower().endswith('.png')]


    # Ensure all markers are loaded correctly
    round_markers = [cv2.imread(file, cv2.IMREAD_UNCHANGED) for file in round_marker_files]
    square_markers = [cv2.imread(file, cv2.IMREAD_UNCHANGED) for file in square_marker_files]


    # Creating the training and holdout dtasets for data development
    print('Generating training data')
    training_data_info = create_dataset_from_images(n_total_images=N_TRAINING_IMAGES,
                                                    num_images_per_bg=1,
                                                    num_background_images=int(N_TRAINING_IMAGES*0.1),
                                                    round_markers=round_markers,
                                                    square_markers=square_markers,
                                                    background_folder=TRAINING_RAW_DATA_PATH,
                                                    output_folder=TRAINING_OUTPUT_PATH,
                                                    random_marker_rotation=RANDOM_MARKER_ROTATION,
                                                    random_image_rotation=RANDOM_IMAGE_ROTATION,
                                                    convert_to_bw=BLACK_AND_WHITE,
                                                    apply_edge_detection=EDGE_TRANSFORM,
                                                    apply_brightness_shift=RANDOM_BRIGHTNESS_SHIFT)

    print('Generating holdout data')
    holdout_data_info = create_dataset_from_images(n_total_images=N_HOLDOUT_IMAGES,
                                                    num_images_per_bg=1,
                                                    num_background_images=int(N_HOLDOUT_IMAGES*0.1),
                                                    round_markers=round_markers,
                                                    square_markers=square_markers,
                                                    background_folder=HOLDOUT_RAW_DATA_PATH,
                                                    output_folder=HOLDOUT_OUTPUT_PATH,
                                                    random_marker_rotation=RANDOM_MARKER_ROTATION,
                                                    random_image_rotation=RANDOM_IMAGE_ROTATION,
                                                    convert_to_bw=BLACK_AND_WHITE,
                                                    apply_edge_detection=EDGE_TRANSFORM,
                                                    apply_brightness_shift=RANDOM_BRIGHTNESS_SHIFT)


    print('Data augmentation is complete')

if __name__ == '__main__':
    main()