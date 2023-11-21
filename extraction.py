import numpy as np
from os import listdir
from descriptors import bitdesc
from paths import CRC_dir, CRC_path
import cv2

def main():
    # List for all the signatures from inages with BiTdec, etc..
    listOflists = list()
    print('Extracting features ....')
    # Loop in path and grab subfolders
    counter = 0
    for CRC_class in CRC_dir:
        print(f'Current folder: {CRC_class}')
        # Grab files from subfolders
        for filename in listdir(CRC_path + CRC_class+'/'):
            counter += 1
            img_name = f'{CRC_path}{CRC_class}/{filename}'
            #print(img_name)
            # Read/Load Image as gray
            img = cv2.imread(img_name, 0)
            features = bitdesc(img) + [CRC_class] + [img_name]
            print(f' Image count: {counter}')
            # Add image features to listOflists
            listOflists.append(features)
    final_array = np.array(listOflists)
    np.save('CRC_signatures_v1.npy', final_array)
    print('Extraction concluded successfully!') 

if __name__ == '__main__':
    main()
   