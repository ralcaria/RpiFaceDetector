import os
from collections import namedtuple

def main():
    """
    The goal of this script is to create a database for the target images (training/validation).

    Format:
    imagepath                       label
    </path/to/image/Reigan.png>     Reigan
    .
    .
    .
    </path/to/image/NotReigan.png>  NotReigan

    """

    colnames= ['imagepath','label']
    ImageGroup = namedtuple('ImageGroup', colnames)
    img_rows = []
    DATASET_TYPE = ('training', 'validation')
    INPUTDIR = os.path.join(os.getcwd(), '..', '..', 'input') #Path to images dir

    #Walk thru images dir
    for curr_path, dirs, files in os.walk(os.path.join(INPUTDIR, 'images')):
        if not dirs: #no subfolders in this directory means contains only images
            target = curr_path.split('/')[-1]
            for _file in files:
                img_rows.append(ImageGroup(os.path.join(curr_path, _file), target))

    #Save the training and validation datasets
    for dataset in DATASET_TYPE:
        with open(os.path.join(INPUTDIR, f'{dataset}_dataset.csv'), 'w') as csvfile:
            csvfile.write(','.join(colnames) + '\n')
            for row in img_rows:
                if dataset in row.imagepath:
                    csvfile.write(','.join(list(row)) + '\n')

if __name__ == '__main__':
    main()
