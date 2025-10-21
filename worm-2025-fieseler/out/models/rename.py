import os

def rename(path):
    contentnames = os.listdir(path)
    filepaths = list()
    dirpaths = list()
    for name in contentnames:
        contentpath = os.path.join(path, name)
        if os.path.isdir(contentpath):
            dirpaths.append(contentpath)
        else:
            filepaths.append(contentpath)

    for filepath in filepaths:
        print(filepath)
        os.rename(filepath, filepath.replace('_no_image_data', ''))

    for dirpath in dirpaths:
        #print(dirpath)
        rename(dirpath)

rename('full')

