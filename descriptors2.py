import cv2
import mahotas.features as ft
from skimage.feature import graycomatrix, graycoprops # scikit-image
from BiT import biodiversity, taxonomy # Bitdesc
import numpy as np
import glob
import os
from typing import List
import pandas as pd
# Define Haralick
def haralick(data):
    all_statistics = ft.haralick(data)
    return all_statistics
def haralick_with_mean(data):
    all_statistics = ft.haralick(data).mean(0)
    return all_statistics

# Define Bitdesc
def bitdesc(data):
    bio = biodiversity(data)
    taxo = taxonomy(data)
    all_statistics = bio + taxo
    return all_statistics

# Gray-Level Co-occurence Matrix
def glcm(data):
    glcm = graycomatrix(data, [2], [0], 256, symmetric=True, normed=True)
    diss = graycoprops(glcm, 'dissimilarity')[0,0]
    cont = graycoprops(glcm, 'contrast')[0,0]
    corr = graycoprops(glcm, 'correlation')[0,0]
    ener = graycoprops(glcm, 'energy')[0,0]
    homo = graycoprops(glcm, 'homogeneity')[0,0]    
    all_statistics = [diss, cont, corr, ener, homo]
    return all_statistics

def Har_BiT_Glcm(data):
    return haralick(data) + bitdesc(data) + glcm(data)


#def convert_class(class_name):
    #if class_name == 'Refuge':
       # return 0
    #elif class_name == 'REFUGE':
        #return 1

def process_images(image_directory, subfolders):
    #container for final set of features
    features = []
    #get list of image

    for type in subfolders:
        print(f'Current folder:{type}')
        #get list of images
        image_paths = glob.glob(os.path.join(image_directory + type, "*"))
        count = 0
         #Read each image in RGB/BGR
        for image_path in image_paths:
            img = cv2.imread(image_path)
             #Diviser image en R, G, B
            B, G, R = cv2.split(img)
            #Convertir les images en grayscale
            RGB_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            R_gray= R
            G_gray = G
            B_gray = B
    # #Extract features haralick + BiT + glcm
            #RGB_feat = np.array(Har_BiT_Glcm(RGB_gray))
            R_feat = np.array(bitdesc(R_gray))
            G_feat = np.array(glcm(G_gray))
            B_feat = np.array(haralick_with_mean(B_gray))
            class_ = np.array([type])

            #if process_images == "REFUGE":
                #cls = ([convert_class(type)])
            #else:
                #cls = ([convert_class2(type)])
        #Concatenate the features
            combined_features = np.hstack((R_feat, G_feat, B_feat, class_))
        #Append to final list
            features.append(combined_features)
            print(f'{count+1}Features extracted.... Folder -> {type}')
            count +=1

    #print(type, image_paths)
    return features

def save_to_csv(features, output_path):
    #convert to dataframe
    df = pd.DataFrame(features)
    #save to csv file
    df.to_csv(output_path, index=False)



def main():
    Refuge_path = r'C:\Users\NaNa\Desktop\Ete2023\IA_2\Projetcovid\Datasets\REFUGE/'
    Refuge_dir:List[str] = os.listdir(Refuge_path)
    output_path = 'refuge_h_bit_glcm_.csv'
    # #process the image
    features = process_images(Refuge_path, Refuge_dir)
    # #save the features to csv file
    save_to_csv(features, output_path)
if __name__ == '__main__':
    main()

   