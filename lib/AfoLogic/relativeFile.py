from os import path

# find a file relative to the file we are in
def find(root: str, directory: str, file_name: str) -> str:
    directory_path = path.join(path.dirname(path.abspath(root)), directory)
    return path.join(directory_path, file_name)

# example: find_file(__file__, '../data', 'test.txt')