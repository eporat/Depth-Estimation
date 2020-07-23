import gzip
import shutil

def gunzip_shutil(source_filepath, dest_filepath, block_size=65536):
    """Unzips and writes to destination file

    Args:
        source_filepath (string): source file path
        dest_filepath (string): destination file path
        block_size (int, optional): block size. Defaults to 65536.
    """
    with gzip.open(source_filepath, 'rb') as s_file, \
            open(dest_filepath, 'wb') as d_file:
        shutil.copyfileobj(s_file, d_file, block_size)
        

