from typing import List
from os import listdir

Refuge_path = r"C:\Users\NaNa\Desktop\Ete2023\IA_2\Projetcovid\Datasets\REFUGE/"
Refuge_dir:List[str] = listdir(Refuge_path)
print(Refuge_path)