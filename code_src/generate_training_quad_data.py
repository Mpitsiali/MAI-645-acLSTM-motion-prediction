import read_bvh
import numpy as np
from os import listdir
import os


def generate_quad_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    if (os.path.exists(tar_traindata_folder) == False):
        os.makedirs(tar_traindata_folder)
    bvh_dances_names = listdir(src_bvh_folder)
    for bvh_dance_name in bvh_dances_names:
        name_len = len(bvh_dance_name)
        if (name_len > 4):
            if (bvh_dance_name[name_len - 4: name_len] == ".bvh"):
                dance = read_bvh.get_quad_train_data(src_bvh_folder + bvh_dance_name)
                np.save(tar_traindata_folder + bvh_dance_name + ".npy", dance)



def generate_bvh_from_quad_traindata(src_train_folder, tar_bvh_folder):
    if (os.path.exists(tar_bvh_folder)==False):
        os.makedirs(tar_bvh_folder)
    dances_names=listdir(src_train_folder)
    for dance_name in dances_names:
        name_len=len(dance_name)
        if(name_len>4):
            if(dance_name[name_len-4: name_len]==".npy"):
                dance=np.load(src_train_folder+dance_name)
                dance2=[]
                for i in range(int(dance.shape[0]/8)):
                    dance2=dance2+[dance[i*8]]
                read_bvh.get_quad_data_from_npy(tar_bvh_folder + dance_name + ".bvh", np.array(dance2))



# standard_bvh_file = "/content/MAI645_final_project/train_data_bvh/standard.bvh"
standard_bvh_file = "train_data_bvh/standard.bvh"
weight_translation = 0.01
skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)

# Encode data from bvh to positional encoding
generate_quad_traindata_from_bvh("train_data_bvh/martial/","train_data_quad/martial/")

# Decode from positional to bvh
generate_bvh_from_quad_traindata("train_data_quad/martial/", "test_data_quad_bvh/martial/",)