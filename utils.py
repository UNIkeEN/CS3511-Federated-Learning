import os
import shutil
import socket

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

def receive_data(sock, buffer_size):
    data = b''
    while True:
        try:
            packet = sock.recv(buffer_size)
            if packet:
                data += packet
                if data[-3:] == b"FIN":  # check if it's finish signal, return "FIN" to shutdown client
                    data = b"FIN"
                    break
                if data[-3:] == b"END":
                    data = data[:-3]  
                    break
            else:
                break
        except socket.error as e:
            print("Socket error while receiving data:", e)
            return None

    return data
