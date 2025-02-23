import os


dir_path = "test"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory created: {dir_path}")
