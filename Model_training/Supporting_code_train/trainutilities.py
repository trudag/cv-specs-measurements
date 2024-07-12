import os
import yaml

def is_file_in_folder(file_path, folder_path):
    # Normalize the paths to ensure consistency
    file_path = os.path.abspath(file_path)
    folder_path = os.path.abspath(folder_path)
    
    # Check if the file is within the folder
    return os.path.commonpath([file_path, folder_path]) == folder_path

def save_yaml_from_string(yaml_string, file_path):
    """
    Saves a YAML string to a file.

    :param yaml_string: String containing YAML content
    :param file_path: Path where the YAML file will be saved
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        file.write(yaml_string)
