#!/usr/bin/env python
# coding: utf-8

#################################################################################
import os
import numpy as np
import pandas as pd 
from glob import glob
import matplotlib.pyplot as plt
import shap
import h5py
import pickle
from joblib import load

models_dir = sorted(glob("../results/newlbls-fu3-espad-fu3-19a-binge-*/*/"))[-1]
models = {}
model_names = list(set([f.split("_")[0] for f in os.listdir(models_dir) if f.split(".")[-1]=="model"]))
for model_name in model_names:
    models.update({model_name: [load(f) for f in glob(models_dir+f"/{model_name}_*.model")]})

    
def get_data(h5_dir):
#     data.keys(), data.attrs.keys()
#     return X, y, X_col_names, group_mask
# X.shape, y.shape, len(X_col_names)
    pass
    
def to_shap(MODEL, X):
    X100 = shap.utils.sample(X, 100) # 100 instances for use as the background distribution
    
    for model_name in models:
        if ( model_name.upper() not in MODEL):
            print("skipping model {}".format(model_name))
            continue
        print("generating SHAP values for model = {} ..".format(model_name))
        for i, model in enumerate(models[model_name]):
            if i!=2:
                print("Skipping model '{}': {}' as it is taking too long".format(model_name, i))
                continue
            if i==2:
                explainer = shap.Explainer(model.predict, X100, output_names=["Healthy","AUD-risk"])
                shap_values = explainer(X)
            if not os.path.isdir("explainers"):
                os.makedirs("explainers")
        
            with open("explainers/{}_holdout_shap.sav".format(model_name+str(i)), "wb") as f:
                pickle.dump(shap_values, f)

def plot_shap(MODEL, DATA, PLOT):
    # 1. choose the model (i = [0:6])
    # 2. subgroup: triaining and holdout
    # 3. plot: summary_plot bar, dot and summary_plot
    pass