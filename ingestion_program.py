import torch
import torch.nn as nn
import numpy as np
import sys
import os
import SimpleITK as sitk
import time
import pickle
import psutil



# Paths
input_dir = '/app/input_data/' # Data
output_dir = '/app/output/'    # For the predictions
program_dir = '/app/program'
submission_dir = '/app/ingested_program' # The code submitted
# input_dir = 'test_dataset_example/image'
# output_dir = 'pred_example'
# program_dir = './'
# submission_dir = './'
sys.path.append(output_dir)
sys.path.append(program_dir)
sys.path.append(submission_dir)

from submission import test

def load_data():  # input: ["1.nii.gz", "2.nii.gz", "3.nii.gz", "4.nii.gz", ...]
    lst = os.listdir(input_dir)
    input_images = {}
    for i in lst:
        input_images[i[:-7] + ".nii.gz"] = os.path.join(input_dir, i)
    return input_images

def main():
    """ The ingestion program.
    """
    print('Ingestion program.')
    start = time.time()
    
    input_images = load_data()
    total_bbox_dic = {}
    print('Data loaded')
    for k, v in input_images.items():
        image = sitk.ReadImage(v)
        pred_bbox, pred_image = test(image)
        total_bbox_dic[k] = pred_bbox
        sitk.WriteImage(pred_image, os.path.join(output_dir, k[:-7] + "pred.nii.gz"))
    pickle.dump(total_bbox_dic, open(os.path.join(output_dir, 'bbox_pred.pkl'), 'wb')) # Save the predictions

    duration = time.time() - start
    print(f'Completed. Total duration: {duration}')

if __name__ == '__main__':
    main()