import os
import shutil

def check_directory(directory):
    """
    Check if the path exists and is empty.
    """
    if os.path.exists(directory):
        if os.listdir(directory):
            response = input(f"The directory {directory} is not empty. Clear all? (y/n): ").strip().lower()
            if response == 'y':
                # clear directory
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Error occurred while deleting {file_path}. Reason: {e}')
    else:
        # If the directory does not exist, create it
        os.makedirs(directory)
