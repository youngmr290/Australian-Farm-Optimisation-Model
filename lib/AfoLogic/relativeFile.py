from os import path

# find a file relative to the file we are in
def find(root: str, directory: str, file_name: str) -> str:
    directory_path = path.join(path.dirname(path.abspath(root)), directory)
    return path.join(directory_path, file_name)

# example: find_file(__file__, '../data', 'test.txt')

def findExcel(file_name: str) -> str:
    return find(__file__, "../../ExcelInputs", file_name)
# this will only work from inside the AfoLogic or RawVersion folders
# as its set to go up exactly two directories