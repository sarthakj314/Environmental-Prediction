import os
import sys
import yaml
import torch
import datetime
import numpy as np
import torchvision.transforms as T
from dateutil.relativedelta import relativedelta
from torch.utils.data import Dataset, DataLoader
from netCDF4 import Dataset as ncDataset
from collections import defaultdict
from tqdm import tqdm

from disc import *

cwd = os.getcwd()
args = sys.argv
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('utils/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

torch.manual_seed(config['seed'])

'''
transform -> Data Augmentation + Preprocessing
Dataset -> Retrives Data + Communicates With DataLoader
display_progress -> Visualize Data During Training
'''

class SentinelDataset():
    def __init__(self, valid_indexes, data_directory, DATES_PATH):
        self.Original_sent2 = ncDataset(os.path.join(data_directory, 'sent2_b1-b4_train.nc'))
        self.Additional_sent2 = ncDataset(os.path.join(data_directory, 'sent2_deforestation_segmentation.nc'))

        # Coordinates
        Original_COORDS = np.array(self.Original_sent2.variables['center_lat_lons']).T
        Additional_COORDS = np.array(self.Additional_sent2.variables['center_lat_lons']).T
        self.COORDS = np.concatenate((Original_COORDS, Additional_COORDS), axis=1).T

        # Dates
        self.DATES = np.load(DATES_PATH)
        self.DATES_SHORT = sorted(np.unique([date[:7] for date in self.DATES]), key=lambda x: datetime.datetime.strptime(x, '%Y-%m'))
        self.DATES_SHORT_DICT = {self.DATES_SHORT[i] : i for i in range(len(self.DATES_SHORT))}

        # Data Bands
        Original_DATA_BANDS = np.array(self.Original_sent2.variables['data_band'])
        Additional_DATA_BANDS = np.array(self.Additional_sent2.variables['data_band'])
        self.Additional_DATA_BANDS_DICT = {Additional_DATA_BANDS[i] : i for i in range(Additional_DATA_BANDS.shape[0])}
        self.DATA_BANDS = Original_DATA_BANDS

        self.COORD_YEAR_MONTH = defaultdict(list)
        for i in valid_indexes:
            date = self.DATES[i]
            self.COORD_YEAR_MONTH[(tuple(self.COORDS[i]), date[:7])].append(i)
        tot = 0
        for key in self.COORD_YEAR_MONTH:
            tot += len(self.COORD_YEAR_MONTH[key])
    
    def get_image(self, idx):
        Original_num_images = self.Original_sent2.variables['images'].shape[0]
        Additional_num_images = self.Additional_sent2.variables['images'].shape[0]
        if idx < Original_num_images:
            return np.array(self.Original_sent2.variables['images'][idx]).astype(np.float32)
        else:
            idx -= Original_num_images
            img = np.array(self.Additional_sent2.variables['images'][idx]).astype(np.float32)
            img = img[[self.Additional_DATA_BANDS_DICT[band] for band in self.DATA_BANDS]]
            return img
        
    def get_date(self, idx):
        return self.DATES[idx]
    
    def get_coords(self, idx):
        return self.COORDS[idx]
    
    def __len__(self):
        return len(self.DATES)
    
    def getitem(self, idx):
        return self.get_image(idx), self.get_date(idx), self.get_coords(idx)

    def get_past_month(self, date, coords, delta):
        current_date = datetime.datetime.strptime(date[:7], '%Y-%m')
        past_date = current_date - relativedelta(months=delta)
        past_date = past_date.strftime('%Y-%m')
        if len(self.COORD_YEAR_MONTH[(tuple(coords), past_date)]) == 0:
            return -1
        else:
            # Randomly Selecting One Image
            return np.random.choice(self.COORD_YEAR_MONTH[(tuple(coords), past_date)])

    def get_past_months(self, date, coords):
        current_date = datetime.datetime.strptime(date[:7], '%Y-%m')
        past_12_months = [current_date - relativedelta(months=i) for i in range(1, 13)][::-1]
        past_12_months = [date.strftime('%Y-%m') for date in past_12_months]
        idxs = []
        for month in past_12_months:
            if len(self.COORD_YEAR_MONTH[(tuple(coords), month)]) == 0:
                idxs.append(-1)
            else:
                # Randomly Selecting One Image
                idxs.append(np.random.choice(self.COORD_YEAR_MONTH[(tuple(coords), month)]))
        dates = []
        for month in past_12_months:
            if month in self.DATES_SHORT_DICT:
                dates.append(self.DATES_SHORT_DICT[month])
            else:
                dates.append(0)
        mask = np.array([idx != -1 for idx in idxs])
        return idxs, dates, mask


class Transformer_Dataset(Dataset):
    def __init__(self, valid_indexes_path, data_directory, dates_path, data_cap = None):
        self.indexes = np.load(valid_indexes_path)
        self.sent2 = SentinelDataset(self.indexes, data_directory, dates_path)
        if data_cap is None:
            self.len = len(self.indexes)
        else:
            self.len = min(len(self.indexes), data_cap)
        print('Total Data (after min):', self.len)
    
    def __len__(self):
        return self.len

    def process(self, img):
        img = img.squeeze()[1:4][::-1].transpose(1, 2, 0)
        percentile = np.percentile(img, [5, 95])
        img = np.clip(img, percentile[0], percentile[1])
        img = (img - percentile[0]) / (percentile[1] - percentile[0])
        img = T.ToTensor()(img)
        return img

    def __getitem__(self, idx):
        idx = self.indexes[idx]
        GT, date, coords = self.sent2.getitem(idx)
        GT = self.process(GT)
        past_idxs, dates, mask = self.sent2.get_past_months(date, coords)
        past_images = [self.process(self.sent2.get_image(idx)) for idx in past_idxs] 
        past_images = torch.stack(past_images)
        dates = torch.tensor(dates)
        mask = torch.tensor(mask)
        return past_images.type(torch.FloatTensor), GT, dates, mask


class Pix2PixHD_Dataset(Dataset):
    def __init__(self, valid_indexes_path, data_directory, dates_path, data_cap = None):
        self.tmp_indexes = np.load(valid_indexes_path)
        self.sent2 = SentinelDataset(self.tmp_indexes, data_directory, dates_path)
        self.indexes = []
        for idx in self.tmp_indexes:
            _, date, coords = self.sent2.getitem(idx)
            if self.sent2.get_past_month(date, coords, 1) != -1:
                self.indexes.append(idx)
        self.indexes = np.array(self.indexes)
        if data_cap is None:
            self.len = len(self.indexes)
        else:
            self.len = min(len(self.indexes), data_cap)
        print('Total Data (after min):', self.len)

    def __len__(self):
        return self.len
    
    def process(self, img):
        img = img.squeeze()[1:4][::-1].transpose(1, 2, 0)
        percentile = np.percentile(img, [5, 95])
        img = np.clip(img, percentile[0], percentile[1])
        img = (img - percentile[0]) / (percentile[1] - percentile[0])
        img = T.ToTensor()(img)
        return img
    
    def __getitem__(self, idx, delta_time = 1):
        # Return GT, input_image, dates
        idx = self.indexes[idx]
        GT, date, coords = self.sent2.getitem(idx)
        GT = self.process(GT)
        past_idx = self.sent2.get_past_month(date, coords, delta_time)
        if past_idx == -1:
            return None
        input_image = self.process(self.sent2.get_image(past_idx))
        date = torch.tensor(self.sent2.get_date(past_idx))
        return input_image, GT, date


'''
SentinelDataset = SentinelDataset(config['data_directory'], config['dates_path'])
valset = Dataset(SentinelDataset, config['val_data_path'], total_data = 10)
valloader = DataLoader(valset, batch_size=10, shuffle=True, num_workers=2)
trainset = Dataset(SentinelDataset, config['train_data_path'], total_data = 20)
trainloader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)
'''

'''
testset = Dataset(SentinelDataset, config['test_data_path'], total_data = 10)
testloader = DataLoader(testset, batch_size=10, shuffle=True, num_workers=2)

for past_images, GT, dates, mask in testloader:
    print(past_images.shape, GT.shape, dates.shape, mask.shape)
    print(dates[0], mask[0])
'''