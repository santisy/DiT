import subprocess

def copy_back_fn(file_path:str, dst:str, isdir=False):
    additional_flag = "-rf " if isdir else ""
    try:
        result = subprocess.run(f"cp {additional_flag}{file_path} {dst}",
                                shell=True,
                                timeout=int(60 * 10))
        if result.returncode == 0:
            print(f"Successfully copy {file_path} to {dst}")
        else:
            print(f"Command failed with return code {result.returncode}")
    except subprocess.TimeoutExpired:
        # Handle the case where the command exceeded the timeout
        print("Copy command execution timed out")
    except Exception as e:
        # Handle other exceptions
        print(f"An error occurred: {e}")