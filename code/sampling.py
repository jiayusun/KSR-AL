from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


from PIL import Image
import os

features_path = '/home/jy/ksral/rev1/feat3' 
w_file = '/home/jy/ksral/rev1/f_clustering_200.txt'   
w2_file = '/home/jy/ksral/rev1/f_clustering_200_zhanbi.txt'   
n_file = '/home/jy/ksral/image_list_2000/zhanbi/c1_name.txt'   
z_file = '/home/jy/ksral/image_list_2000/zhanbi/c1_zhanbi.txt'    
doc = open(w_file,'w')
doc2 = open(w2_file,'w')
times = 0



name_list = []
for lines in open(n_file, 'r'):
    feature_name = ''.join(['feat_', lines[:-5], '.txt'])
    name_list.append(feature_name)

zhanbi_list = []
for lines in open(z_file, 'r'):
    zhanbi_list.append(lines[:-1])

list1 = []

for filename in name_list:
    print(filename)
    data = []
    listStr = [features_path, '/', filename]
    file_full_name = ''.join(listStr)
    for lines in open(file_full_name, 'r'):
        data.append(float(lines[:-1]))
    list1.append(data)

x = np.asarray(list1)
centroid = DBSCAN(eps=0.05, min_samples=1).fit_predict(x)    
list2 = []
list3 = []
size = max(centroid)-min(centroid)
for j in range(size):                
    tmp = np.where(centroid == j)
    list2.append(tmp[0])
    if tmp[0].size <= 1:
        list3.append(tmp[0][0])
    else:
        choice1 = tmp[0]
        list3.append(choice1[0])
list4 = []
for jj in list3:
    list4.append(name_list[jj])
    final_name = name_list[jj][5:]
    print(final_name)
    doc.write(name_list[jj][5:] + '\n')
doc.close()
list5 = []
for jj in list3:
    list5.append(zhanbi_list[jj])
    final_zhanbi = zhanbi_list[jj]
    print(final_zhanbi)
    doc2.write(zhanbi_list[jj] + '\n')

doc2.close()
