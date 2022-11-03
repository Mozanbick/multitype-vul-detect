"""
version: 2022/11/2
author: zjh

This script is going to re-arrange the sard dataset folder structure::

the original folder structure maybe like this:
    root folder
        - 000 (first level folder)
            -- 001 (second level folder)
                --- basic-01.c (file1)
                --- basic-02.c (file2)
            -- 017
                --- CWE-119_filename_func_ver.type.c (given file name)
                --- ...
            ...
        - 001
            ...
        ...

Since files in each second level folder forms a test case, and function calls are limited in one test case,
we combine the first level folder name and the second level folder name to get the identifier of each test case.

In addition, we add the test case information, as well as the function start line number (since we need to extract
the function instead of a whole C file) and the hash token (since one file could extract multiple functions), to
the filename.
"""
