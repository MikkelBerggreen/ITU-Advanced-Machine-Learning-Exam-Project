# IMPORTS
import datetime
import os
import pickle
import tarfile
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models, transforms  # Import torchvision and transforms
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights, AlexNet_Weights
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset