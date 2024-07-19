# helpful util files that are used in the project

def write_to_file(file_path: str, text: str):
    """helper function to write content to a file
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

def read_from_file(file_path):
    """helper function to read content from a file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()