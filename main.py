import sys, getopt
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from train import train_CNN_CBAM_Model, train_ViT_Model
from performancemetrics import ROC_with_CI_outcomes
import argparse

def main(args):
    if not args.inputfile:
        print('Provide the input file: -i')
        sys.exit(2)
    data = pd.read_csv(args.inputfile)
    if args.model == 'ViT':
        true_variables, pred_variables, outcomes, models = train_ViT_Model(data)
    elif args.model == 'CNN':
        true_variables, pred_variables, outcomes, models = train_CNN_CBAM_Model(data)
    else:
        print('Invalid model choice. Choose between "ViT" or "CNN".')
        sys.exit(2)
    
    ROC_with_CI_outcomes(true_variables, pred_variables, outcomes, args.model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile')
    parser.add_argument('-m', '--model', choices=['ViT', 'CNN'], help='Choose between ViT or CNN model')
    args = parser.parse_args()
    main(args)

    