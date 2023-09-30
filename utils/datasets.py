import os
import gc
import json
import math
from time import mktime
import numpy as np
import polars as pl
from pathlib import Path
from datetime import timedelta
from datetime import datetime
from datetime import time
from datetime import date
from dateutil.parser import parse
from utils.npy import NpyFileAppend

from utils.general import yaml_load
from utils.general import list_convert
from utils.general import list_uniqifier
from utils.visuals import progress_bar

# from sklearn.preprocessing import MinMaxScaler

class DatasetController():
    def __init__(self, 
                 configsPath=None, 
                 resample=5, 
                 # startTimeId=0, 
                 workers=8, 
                 splitRatio=(0.7, 0.2, 0.1), 
                 lag=5, 
                 ahead=1, 
                 offset=1, 
                 savePath='.', 
                 filling=None, 
                 low_memory=False,
                 normalization=False,
                 cyclicalPattern=False,
                 skip_test=False):
        """ Read data config """
        if isinstance(configsPath, dict):
            self.dataConfigs = configsPath
        else:
            self.dataConfigs = yaml_load(configsPath)

        try:
            self.dataPaths = list_convert(self.dataConfigs['path'])
            self.dateFeature = self.dataConfigs['date']
            self.targetFeatures = list_convert(self.dataConfigs['target'])
            self.delimiter = self.dataConfigs['delimiter']
            self.trainFeatures = list_convert(self.dataConfigs['features'])
            self.segmentFeature = self.dataConfigs['segmentFeature']
            self.timeFormat = self.dataConfigs['timeFormat']
            self.granularity = self.dataConfigs['granularity']
            self.hasHeader = self.dataConfigs['hasHeader']
            self.fileNameAsFeature = self.dataConfigs['fileNameAsFeature']
            self.stripTime   = self.dataConfigs['stripTime']
            
            self.xtrain = np.array([])
            self.ytrain = np.array([])
            self.xval = np.array([])
            self.yval = np.array([])
            self.xtest = np.array([])
            self.ytest = np.array([])

            self.trainraw = np.array([])
            self.valraw = np.array([])
            self.testraw = np.array([])
        except KeyError:
            mmap_mode = 'r' if low_memory else None
            self.xtrain = np.load(file=self.dataConfigs['xtrain'], mmap_mode=mmap_mode)
            self.ytrain = np.load(file=self.dataConfigs['ytrain'], mmap_mode=mmap_mode)
            self.xval = np.load(file=self.dataConfigs['xval'], mmap_mode=mmap_mode)
            self.yval = np.load(file=self.dataConfigs['yval'], mmap_mode=mmap_mode)
            self.xtest = np.load(file=self.dataConfigs['xtest'], mmap_mode=mmap_mode)
            self.ytest = np.load(file=self.dataConfigs['ytest'], mmap_mode=mmap_mode)

        self.workers = workers
        self.resample = resample
        self.splitRatio = splitRatio
        self.lag = lag
        self.ahead = ahead
        self.offset = offset
        self.low_memory = low_memory
        self.savePath = savePath

        self.dataFilePaths = []
        self.df = pl.DataFrame()

        self.num_samples = []
        # self.smoothing = False
        self.filling = filling
        self.normalization = normalization
        self.scaler = {}
        self.cyclicalPattern = cyclicalPattern
        self.skip_test = skip_test

    def execute(self):
        if len(self.ytrain) != 0: return self
        self.GetDataPaths(dataPaths=self.dataPaths, storeValues=True)
        self.ReadFileAddFetures(csvs=self.dataFilePaths, 
                                hasHeader=self.hasHeader,
                                separator=self.delimiter,
                                tryParseDates=True,
                                lowMemory=self.low_memory, 
                                fileNameAsFeature=self.fileNameAsFeature,
                                storeValues=True)
        # self.df = self.df.with_columns(pl.from_epoch("OPEN_TIME", time_unit="ms"))
        # print(self.df)
        # exit()
        self.df = self.df.drop_nulls()
        self.df = self.df.unique()
        sort_by = [self.segmentFeature, self.dateFeature] if self.segmentFeature is not None else [self.dateFeature]
        self.df = self.df.unique(subset=sort_by, maintain_order=True)
        if self.dateFeature is not None: self.df = self.df.with_columns(pl.col(self.dateFeature).cast(pl.Datetime))
        if self.stripTime is not None: self.df = self.StripDataset(df=self.df, dateColumn=self.dateFeature, stripTime=self.stripTime)
        self.ReduceMemoryUsage(df=self.df, info=False, inplace=True)

        """ Cyclical Pattern """
        if self.cyclicalPattern: 
            print('To be implemented'); exit()
            self.ReduceMemoryUsage(df=self.df, info=False, inplace=True)
        
        """ Get used columns """
        used_cols = [self.dateFeature, self.segmentFeature, *self.trainFeatures, *self.targetFeatures]
        # print(used_cols)
        if self.segmentFeature is None: used_cols.remove(self.segmentFeature)
        # print(used_cols)
        self.df = self.df[list_uniqifier(used_cols)]


        """ Splitting Dataset """
        save_dir = self.savePath
        save_dir = Path(save_dir) / 'values'
        save_dir.mkdir(parents=True, exist_ok=True) 
        if self.segmentFeature:
            if self.dateFeature: self.df = self.df.sort(by=[self.segmentFeature, self.dateFeature])
            else: self.df = self.df.sort(by=[self.segmentFeature])
        # print(self.df.with_columns(pl.col('CO2 (ppm)').cast(pl.Float64)))
        self.SplittingData(self.df,
                           splitRatio=self.splitRatio,
                           lag=self.lag, 
                           ahead=self.ahead, 
                           offset=self.offset,
                           trainFeatures=self.trainFeatures,
                           targetFeatures=self.targetFeatures,
                           segmentFeature=self.segmentFeature,
                           dateFeature=self.dateFeature, 
                           lowMemory=self.low_memory, 
                           saveDir=save_dir, 
                           storeValues=True,
                           granularity=self.granularity,
                           # saveRaw=self.normalization is not None)  
                           saveRaw=self.normalization)  

        """ Filling Missing Data """
        if self.filling: print('To be implemented!'); exit()   

        # exit()
        """ Normalization """
        # if self.normalization: 
        if self.normalization is not None: 
            print('Normalizing Data')
            for col in self.trainraw.columns:
                if col == self.dateFeature: continue
                self.scaler[col] = {
                    'min': self.trainraw[col].min(),
                    'max': self.trainraw[col].max(),
                    'mean': self.trainraw[col].mean(),
                    'median': self.trainraw[col].median(),
                    'std': self.trainraw[col].std(),
                    'variance': self.trainraw[col].var(),
                    'quantile_25': self.trainraw[col].quantile(quantile=0.25, interpolation='nearest'),
                    'quantile_50': self.trainraw[col].quantile(quantile=0.5, interpolation='nearest'),
                    'quantile_75': self.trainraw[col].quantile(quantile=0.75, interpolation='nearest'),
                    'iqr': self.trainraw[col].quantile(quantile=0.75, interpolation='nearest') - self.trainraw[col].quantile(quantile=0.25, interpolation='nearest'),
                    'method': self.normalization
                }
                # self.df = self.df.with_columns((pl.col(col) - self.scaler[col]['min']) / (self.scaler[col]['max'] - self.scaler[col]['min']))
                if self.normalization == 'minmax':
                    self.df = self.df.with_columns((pl.col(col) - self.scaler[col]['min']) / (self.scaler[col]['max'] - self.scaler[col]['min']))
                elif self.normalization == 'standard':
                    self.df = self.df.with_columns((pl.col(col) - self.scaler[col]['mean']) / self.scaler[col]['std'])
                elif self.normalization == 'robust':
                    self.df = self.df.with_columns((pl.col(col) - self.scaler[col]['median']) / (self.scaler[col]['iqr'] + 1e-12))
                    
            self.scaler = self.scaler[self.targetFeatures[0]]
            
            with open(save_dir / 'scaler.json', "w") as json_file:
                # print('='*123)
                json.dump(self.scaler, json_file)

            self.SplittingData(self.df,
                           splitRatio=self.splitRatio,
                           lag=self.lag, 
                           ahead=self.ahead, 
                           offset=self.offset,
                           trainFeatures=self.trainFeatures,
                           targetFeatures=self.targetFeatures,
                           segmentFeature=self.segmentFeature,
                           dateFeature=self.dateFeature, 
                           lowMemory=self.low_memory, 
                           saveDir=save_dir, 
                           storeValues=True,
                           granularity=self.granularity,
                           saveRaw=False)  
            #     self.trainraw = self.trainraw.with_columns((pl.col(col) - self.scaler[col]['min']) / (self.scaler[col]['max'] - self.scaler[col]['min']))
            #     self.valraw = self.valraw.with_columns((pl.col(col) - self.scaler[col]['min']) / (self.scaler[col]['max'] - self.scaler[col]['min']))
            #     self.testraw = self.testraw.with_columns((pl.col(col) - self.scaler[col]['min']) / (self.scaler[col]['max'] - self.scaler[col]['min']))

            # # TODO: case more than 1 target features
            # self.scaler = self.scaler[self.targetFeatures[0]]
            # print('\t- Train')
            # train = self.SplittingData(self.trainraw,
            #                            splitRatio=(1, 0, 0),
            #                            lag=self.lag, 
            #                            ahead=self.ahead, 
            #                            offset=self.offset,
            #                            trainFeatures=self.trainFeatures,
            #                            targetFeatures=self.targetFeatures,
            #                            segmentFeature=self.segmentFeature,
            #                            dateFeature=self.dateFeature, 
            #                            lowMemory=self.low_memory, 
            #                            saveDir=save_dir, 
            #                            storeValues=False,
            #                            granularity=self.granularity,
            #                            saveRaw=False,
            #                            subname='norm') 
            # self.xtrain = train[0]
            # self.ytrain = train[1]
            # del train
            # gc.collect()

            # print('\t- Val')
            # val = self.SplittingData(self.valraw,
            #                          splitRatio=(1, 0, 0),
            #                          lag=self.lag, 
            #                          ahead=self.ahead, 
            #                          offset=self.offset,
            #                          trainFeatures=self.trainFeatures,
            #                          targetFeatures=self.targetFeatures,
            #                          segmentFeature=self.segmentFeature,
            #                          dateFeature=self.dateFeature, 
            #                          lowMemory=self.low_memory, 
            #                          saveDir=save_dir, 
            #                          storeValues=False,
            #                          granularity=self.granularity,
            #                          saveRaw=False,
            #                          subname='norm') 
            # self.xval = val[0]
            # self.yval = val[1]
            # del val
            # gc.collect()

            # print('\t- Test')
            # test = self.SplittingData(self.testraw,
            #                           splitRatio=(1, 0, 0),
            #                           lag=self.lag, 
            #                           ahead=self.ahead, 
            #                           offset=self.offset,
            #                           trainFeatures=self.trainFeatures,
            #                           targetFeatures=self.targetFeatures,
            #                           segmentFeature=self.segmentFeature,
            #                           dateFeature=self.dateFeature, 
            #                           lowMemory=self.low_memory, 
            #                           saveDir=save_dir, 
            #                           storeValues=False,
            #                           granularity=self.granularity,
            #                           saveRaw=False,
            #                           subname='norm') 
            # self.xtest = test[0]
            # self.ytest = test[1]
            # del test
            # gc.collect()
        return self
    
    def GetDataPaths(self, 
                     dataPaths:list, 
                     extensions:tuple=('.csv'), 
                     description:str=' Getting files', 
                     storeValues:bool=False) -> list[str] | None:
        if not isinstance(dataPaths, list): dataPaths = [dataPaths]
        dataFilePaths = []
        with progress_bar() as progress:
            for path in progress.track(dataPaths, description=description):
                if os.path.isdir(path): [dataFilePaths.append(Path(root, file)) for root, _, files in os.walk(path) for file in files if file.endswith(extensions)]
                elif path.endswith(extensions) and os.path.exists(path): dataFilePaths.append(path)
        assert len(dataFilePaths) > 0, 'No csv file(s)'
        dataFilePaths = [os.path.abspath(csv) for csv in list_convert(dataFilePaths)]
        if storeValues: 
            if len(self.dataFilePaths) > 0: self.dataFilePaths.extend(dataFilePaths)  
            else: self.dataFilePaths = dataFilePaths 
        else:
            return dataFilePaths

    def ReadFileAddFetures(self, 
                           csvs:list, 
                           hasHeader:bool=True,
                           separator:str=',',
                           inferSchemaLength:int|None=10_000, 
                           tryParseDates:bool=True,
                           lowMemory:bool=True,
                           fileNameAsFeature:str=None,
                           description:str='  Reading data',
                           storeValues:bool=False) -> pl.DataFrame | None:
        if not isinstance(csvs, list): csvs = [csvs] 
        csvs = [os.path.abspath(csv) for csv in csvs]  

        allData = []
        with progress_bar() as progress:
            for csv in progress.track(csvs, description=description):
                data = pl.read_csv(source=csv, 
                                   separator=separator, 
                                   has_header=hasHeader, 
                                   try_parse_dates=tryParseDates, 
                                   low_memory=lowMemory, 
                                   infer_schema_length=inferSchemaLength)
                if fileNameAsFeature is not None: 
                    d = '.'.join(os.path.basename(os.path.abspath(csv)).split('.')[:-1])
                    if d.isdigit():
                        data = data.with_columns(pl.lit(int(d)).alias(fileNameAsFeature))
                    else:
                        # TODO: handle the case where values are categorical not integers
                        print('To be implemented!!!')
                        exit()
                allData.append(data)
        df = pl.concat(allData)
        if storeValues: self.df = pl.concat([self.df, df])
        else: return df

    def StripDataset(self, 
                     dateColumn:str,
                     df:pl.DataFrame, 
                     stripTime: dict) -> pl.DataFrame:
        if '%Y' in stripTime['format'] and '%H' in self.stripTime['format']: f = datetime
        elif '%Y' in stripTime['format']: f = date
        elif '%H' in stripTime['format']: f = time

        df = df.filter(
                    (pl.col(dateColumn).dt.strftime(stripTime['format']) >= f(*stripTime['start']).strftime(stripTime['format'])) &
                    (pl.col(dateColumn).dt.strftime(stripTime['format']) <= f(*stripTime['end']).strftime(stripTime['format']))
                )
        return df

    def FillDate(self, 
                 df:pl.DataFrame,  
                 dateColumn:str, 
                 low=None, 
                 high=None, 
                 granularity:int=5,
                 storeValues:bool=False): 
        if not low: low=df[dateColumn].min()
        if not high: high=df[dateColumn].max()

        df = df.join(other=pl.date_range(low=low,
                           high=high,
                           interval=timedelta(minutes=granularity),
                           closed='both',
                           name=dateColumn).to_frame(), 
                     on=dateColumn, 
                     how='outer')
        if storeValues: self.df = df
        else: return df
    
    def ReduceMemoryUsage(self, 
                          df:pl.DataFrame, 
                          info:bool=False,
                          inplace:bool=False):
        before = round(df.estimated_size('gb'), 4)
        Numeric_Int_types = [pl.Int8,pl.Int16,pl.Int32,pl.Int64]
        Numeric_Float_types = [pl.Float32,pl.Float64]    
        for col in df.columns:
            col_type = df[col].dtype
            c_min = df[col].min()
            c_max = df[col].max()
            if col_type in Numeric_Int_types:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df = df.with_columns(df[col].cast(pl.Int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df = df.with_columns(df[col].cast(pl.Int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df = df.with_columns(df[col].cast(pl.Int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df = df.with_columns(df[col].cast(pl.Int64))
            elif col_type in Numeric_Float_types:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df = df.with_columns(df[col].cast(pl.Float32))
                else:
                    pass
            elif col_type == pl.Utf8:
                df = df.with_columns(df[col].cast(pl.Categorical))
            else:
                pass
        df = df.shrink_to_fit()
        if info: print(f"Memory usage: {before} GB => {round(df.estimated_size('gb'), 4)} GB")
        if inplace: self.df = df
        else: return df
    
    def GetData(self, shuffle):
        if shuffle:
            np.random.shuffle(self.xtrain)
            np.random.shuffle(self.ytrain)
            np.random.shuffle(self.xval)
            np.random.shuffle(self.yval)
            np.random.shuffle(self.xtest)
            np.random.shuffle(self.ytest)
        return self.xtrain, self.ytrain, self.xval, self.yval, self.xtest, self.ytest 

    def TimeEncoder(self, df):
        day = 24 * 60 * 60 # Seconds in day  
        year = (365.2425) * day # Seconds in year

        # df = self.FillDate(df=df)
        unix = df[self.dateFeature].to_frame().with_columns(pl.col(self.dateFeature).cast(pl.Utf8).alias('unix_str'))
        unix_time = [mktime(parse(t).timetuple()) for t in unix['unix_str'].to_list()]
        df = df.with_columns(pl.lit(unix_time).alias('unix_time'))

        if len(set(df[self.dateFeature].dt.day().to_list())) > 1:
            df = df.with_columns(np.cos((pl.col('unix_time')) * (2 * np.pi / day)).alias('day_cos'))
            df = df.with_columns(np.sin((pl.col('unix_time')) * (2 * np.pi / day)).alias('day_sin'))
            self.trainFeatures.extend(['day_cos', 'day_sin'])
        if len(set(df[self.dateFeature].dt.month().to_list())) > 1:
            df = df.with_columns(np.cos((pl.col('unix_time')) * (2 * np.pi / year)).alias('month_cos'))
            df = df.with_columns(np.sin((pl.col('unix_time')) * (2 * np.pi / year)).alias('month_sin'))
            self.trainFeatures.extend(['month_cos', 'month_sin'])
        
        return df

    def CyclicalPattern(self):
        # assert self.dateFeature is not None
        # if self.segmentFeature:
        #     if self.dateFeature: self.df = self.df.sort(by=[self.segmentFeature, self.dateFeature])
        #     else: self.df = self.df.sort(by=[self.segmentFeature])
        
        # if self.segmentFeature:
        #     dfs = None
        #     for ele in self.df[self.segmentFeature].unique():
        #         df = self.df.filter(pl.col(self.segmentFeature) == ele).clone()
        #         df = self.TimeEncoder(df=df)
        #         if dfs is None: dfs = df
        #         else: dfs = pl.concat([dfs, df])
        #     self.df = dfs.drop_nulls()
        # else: 
        #     self.df = self.TimeEncoder(df=self.df).drop_nulls()
        # self.df = self.TimeEncoder(df=self.df)
        # pass
        self.df = self.df.with_columns(
                    pl.col(self.dateFeature).dt.weekday().alias('weekday'),
                    pl.col(self.dateFeature).dt.month().alias('month'),
                    ((pl.col(self.dateFeature).dt.hour()*60 + pl.col(self.dateFeature).dt.minute())/self.resample).alias(f'{self.resample}mins_hour'),
                )
        self.df = self.df.with_columns(
            ((pl.col(f'{self.resample}mins_hour')/(24*(60/self.resample))*2*np.pi).sin()).alias('hour_sin'), # 0 to 23 -> 23h55
            ((pl.col(f'{self.resample}mins_hour')/(24*(60/self.resample))*2*np.pi).cos()).alias('hour_cos'), # 0 to 23 -> 23h55
            ((pl.col('weekday')/(7)*2*np.pi).sin()).alias('day_sin'), # 1 - 7
            ((pl.col('weekday')/(7)*2*np.pi).cos()).alias('day_cos'), # 1 -  7
            ((pl.col('month')/(12)*2*np.pi).sin()).alias('month_sin'), # 1 -12
            ((pl.col('month')/(12)*2*np.pi).cos()).alias('month_cos') # 1-12
        )
        self.trainFeatures[1:1] = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']

    def _SplitCore(self, 
                   df, 
                   dateFeature, 
                   granularity, 
                   trainFeatures, 
                   targetFeatures,
                   lagWindow, 
                   lag,
                   ahead,
                   splitRatio,
                   segmentFeature=None, 
                   segmentFeatureEle=None):
        trainFeaturesAdded = []
        df = self.FillDate(df=df,
                          dateColumn=dateFeature,
                          granularity=granularity)
        if segmentFeature is not None and segmentFeatureEle is not None: df = df.with_columns(pl.col(segmentFeature).fill_null(pl.lit(segmentFeatureEle)))
        for f in list_uniqifier([*trainFeatures, *targetFeatures]):
            temp = []
            for l in range(1, lagWindow):
                df = df.with_columns(pl.col(f).shift(periods=l).alias(f'{f}_lag_{l}'))
                temp.append(f'{f}_lag_{l}')
            trainFeaturesAdded.append(temp)

        # Get used columns
        trainFeaturesAdded = np.array(trainFeaturesAdded)
        trainColumns = np.transpose(trainFeaturesAdded[:, -lag:]).flatten()
        targetColumns = [*targetFeatures,
                         *[elem_b for elem_b in np.transpose(trainFeaturesAdded[:, :(ahead-1)]).flatten() if any(elem_a in elem_b for elem_a in targetFeatures)]]
        trainColumns = [item for sublist in [trainColumns[i:i + len([*trainFeatures, *targetFeatures])] for i in range(0, len(trainColumns), len([*trainFeatures, *targetFeatures]))][::-1] for item in sublist]
        targetColumns = [item for sublist in [targetColumns[i:i + len(targetFeatures)] for i in range(0, len(targetColumns), len(targetFeatures))][::-1] for item in sublist]
        
        # Drop nulls
        # print(f'{trainColumns = }')
        # print(f'{targetColumns = }')
        # print(f'{[*targetColumns, *targetColumns] = }')
        df = df[[dateFeature, *trainColumns, *targetColumns]].drop_nulls()

        # Get train, val, test idx for splitting
        num_samples = len(df)
        train_idx = math.ceil(num_samples*splitRatio[0])
        val_idx = int(num_samples*(splitRatio[0] + splitRatio[1]))

        x = df[trainColumns].to_numpy()
        x = x.reshape(x.shape[0], int(x.shape[1]/len(trainFeaturesAdded)), len(trainFeaturesAdded))
        y = df[targetColumns].to_numpy()
        # print(f'{y.shape = }')
        if y.shape[0]!=1 or y.shape[1]==1: y = np.squeeze(y)
        # print(f'{y.shape = }')
        # print('=' * 100)
        if y.shape == (): y = np.reshape(y, (1,))

        # print('=======================')
        # print(f'{num_samples = }')
        # print(f'{splitRatio[0] = }')
        # print(f'{splitRatio[1] = }')

        return x, y, train_idx, val_idx, trainFeaturesAdded, [df[:train_idx][dateFeature].max(),
                                                              df[train_idx:val_idx][dateFeature].max(),
                                                              df[train_idx:val_idx][dateFeature].min(),
                                                              df[val_idx:][dateFeature].min()]

    # def _SplitCore(self, 
    #                df, 
    #                dateFeature, 
    #                granularity, 
    #                trainFeatures, 
    #                targetFeatures,
    #                lagWindow, 
    #                lag,
    #                ahead,
    #                splitRatio,
    #                segmentFeature=None, 
    #                segmentFeatureEle=None):
    #     trainFeaturesAdded = []
    #     df = self.FillDate(df=df,
    #                       dateColumn=dateFeature,
    #                       granularity=granularity)
    #     df_clone = df.clone()
    #     if segmentFeature is not None and segmentFeatureEle is not None: df = df.with_columns(pl.col(segmentFeature).fill_null(pl.lit(segmentFeatureEle)))
    #     for f in list_uniqifier([*trainFeatures, *targetFeatures]):
    #         temp = []
    #         for l in range(1, lagWindow):
    #             df = df.with_columns(pl.col(f).shift(periods=-l).alias(f'{f}_lag_{l}'))
    #             # print(df)
    #             # exit()
    #             temp.append(f'{f}_lag_{l}')
    #         trainFeaturesAdded.append(temp)
    #     df = df.drop_nulls()
    #     pl.Config.set_tbl_rows(100)
    #     # pl.Config.set_tbl_cols(20)
    #     # print(df)
    #     df = df.drop_nulls()
    #     num_samples = len(df)
    #     train_idx = math.ceil(num_samples*splitRatio[0])
    #     val_idx = int(num_samples*(splitRatio[0] + splitRatio[1]))
    #     # print('=======================================')
    #     # print(df[train_idx:val_idx][dateFeature].max())
    #     # print(df[train_idx:val_idx][dateFeature].min())
    #     # print(df[train_idx:val_idx])
    #     # print(df.filter((pl.col(dateFeature) <= df[train_idx:val_idx][dateFeature].max()) & (pl.col(dateFeature) >= df[train_idx:val_idx][dateFeature].min())))

    #     trainFeaturesAdded = np.array(trainFeaturesAdded)
    #     trainColumns = [*trainFeatures, *targetFeatures, *np.transpose(trainFeaturesAdded[:, :lag-1]).flatten()]
    #     targetColumns = [elem_b for elem_b in np.transpose(trainFeaturesAdded[:, -ahead:]).flatten() if any(elem_a in elem_b for elem_a in targetFeatures)]
    #     # print(trainColumns)
    #     # print(targetColumns)
    #     # exit()
    #     x = df[trainColumns].to_numpy()
    #     # print(x)
    #     x = x.reshape(x.shape[0], int(x.shape[1]/len(trainFeaturesAdded)), len(trainFeaturesAdded))
    #     # print(x)
    #     # exit()
    #     y = np.squeeze(df[targetColumns].to_numpy())
    #     if y.shape == (): y = np.reshape(y, (1,))

    #     trainFeaturesAdded = []
    #     for f in list_uniqifier([*trainFeatures, *targetFeatures]):
    #         temp = []
    #         for l in range(1, lagWindow):
    #             df = df_clone.with_columns(pl.col(f).shift(periods=l).alias(f'{f}_lag_{l}'))
    #             temp.append(f'{f}_lag_{l}')
    #         trainFeaturesAdded.append(temp)
    #     num_samples = len(df)
    #     train_idx = math.ceil(num_samples*splitRatio[0])
    #     val_idx = int(num_samples*(splitRatio[0] + splitRatio[1]))

    #     trainFeaturesAdded = np.array(trainFeaturesAdded)
    #     trainColumns = np.transpose(trainFeaturesAdded[:, -lag:]).flatten()
    #     targetColumns = [*targetFeatures,
    #                      *[elem_b for elem_b in np.transpose(trainFeaturesAdded[:, :(ahead-1)]).flatten() if any(elem_a in elem_b for elem_a in targetFeatures)]]

    #     x = df[trainColumns].to_numpy()
    #     x = x.reshape(x.shape[0], int(x.shape[1]/len(trainFeaturesAdded)), len(trainFeaturesAdded))
    #     y = np.squeeze(df[targetColumns].to_numpy())
    #     if y.shape == (): y = np.reshape(y, (1,))
    #     # exit()
    #     return x, y, train_idx, val_idx, trainFeaturesAdded, [df[:train_idx][dateFeature].max(),
    #                                                           df[train_idx:val_idx][dateFeature].max(),
    #                                                           df[train_idx:val_idx][dateFeature].min(),
    #                                                           df[val_idx:][dateFeature].min()]

    def SplittingData(self, 
                      df,
                      splitRatio, 
                      lag, 
                      ahead, 
                      offset,
                      trainFeatures:list,
                      targetFeatures:list,
                      segmentFeature:str,
                      dateFeature:str, 
                      lowMemory:bool=False, 
                      saveDir:Path=None, 
                      storeValues:bool=True,
                      granularity:int=1440,
                      saveRaw:bool=False,
                      subname:str=''):

        lag_window = lag + offset

        xtrain = []
        ytrain = []
        xval = []
        yval = []
        xtest = []
        ytest = []

        if saveRaw:
            trainraw = []
            valraw = []
            testraw = []
            header_written = False

        save_file = {
            'invalid': {
                'path': saveDir / 'invalid.txt',
                'writer': open(saveDir / 'invalid.txt', 'a')
                }
        }

        if lowMemory:
            save_file['xtrain'] = {
                'path': saveDir / f'xtrain{subname}.npy',
                'writer': NpyFileAppend(filename=saveDir / f'xtrain{subname}.npy', delete_if_exists=True)
            }
            save_file['ytrain'] = {
                'path': saveDir / f'ytrain{subname}.npy',
                'writer': NpyFileAppend(filename=saveDir / f'ytrain{subname}.npy', delete_if_exists=True)
            }
            save_file['xval'] = {
                'path': saveDir / f'xval{subname}.npy',
                'writer': NpyFileAppend(filename=saveDir / f'xval{subname}.npy', delete_if_exists=True)
            }
            save_file['yval'] = {    
                'path': saveDir / f'yval{subname}.npy',
                'writer': NpyFileAppend(filename=saveDir / f'yval{subname}.npy', delete_if_exists=True)
            }
            save_file['xtest'] = {
                'path': saveDir / f'xtest{subname}.npy',
                'writer': NpyFileAppend(filename=saveDir / f'xtest{subname}.npy', delete_if_exists=True)
            }
            save_file['ytest'] = {
                'path': saveDir / f'ytest{subname}.npy',
                'writer': NpyFileAppend(filename=saveDir / f'ytest{subname}.npy', delete_if_exists=True)
            }
            if saveRaw:
                save_file['trainraw'] = {
                    'path': saveDir / 'trainraw.csv',
                    'writer': None
                }
                save_file['valraw'] = {
                    'path': saveDir / 'valraw.csv',
                    'writer': None
                }
                save_file['testraw'] = {
                    'path': saveDir / 'testraw.csv',
                    'writer': None
                }

        if segmentFeature is not None:
            with progress_bar() as progress:
                for ele in progress.track(df[segmentFeature].unique(), description='Splitting data'):
                    c = df.filter(pl.col(segmentFeature) == ele).clone()
                    d = c.sort(dateFeature)

                    x, y, train_idx, val_idx, trainFeaturesAdded, thething = self._SplitCore(df=d, 
                                                                                   dateFeature=dateFeature, 
                                                                                   granularity=granularity,
                                                                                   trainFeatures=trainFeatures, 
                                                                                   targetFeatures=targetFeatures,
                                                                                   lagWindow=lag_window, 
                                                                                   lag=lag,
                                                                                   ahead=ahead,
                                                                                   splitRatio=splitRatio,
                                                                                   segmentFeature=segmentFeature,
                                                                                   segmentFeatureEle=ele)


                    if any([x[:train_idx].size == 0, 
                            y[:train_idx].size == 0, 
                            (x[train_idx:val_idx].size == 0) & (splitRatio[1] != 0), 
                            (y[train_idx:val_idx].size == 0) & (splitRatio[1] != 0), 
                            (x[val_idx:].size == 0) & (splitRatio[2] != 0), 
                            (y[val_idx:].size == 0) & (splitRatio[2] != 0)]):
                        save_file['invalid']['writer'].write(f'{ele}\n')
                        # if subname == 'norm': print(f'{ele = }')
                        continue
                    # if subname == 'norm': print(f'accept: {ele = }')
                    if saveRaw: 
                        # print(d[:train_idx][dateFeature].max())
                        train = c.filter(pl.col(dateFeature) <= thething[0])
                        val = c.filter((pl.col(dateFeature) <= thething[1]) & 
                                       (pl.col(dateFeature) >= thething[2]))
                        test = c.filter(pl.col(dateFeature) >= thething[3])
                        # print(f'{x.shape = }')
                        # print(f'{y.shape = }')
                        # print(f'{train_idx = }')
                        # print(f'{val_idx = }')
                        # print('=======================')
                        # print(f'{train.shape = }')
                        # print(f'{val.shape = }')
                        # print(f'{test.shape = }')
                        # print(train)
                        # print(test)
                        # exit()
                        # print(train)
                        # print(d[val_idx:][dateFeature].min())
                        # print(d[dateFeature].max())
                        # exit()
                        if lowMemory:
                            train.to_pandas().to_csv(path_or_buf=save_file['trainraw']['path'], mode='a', index=False, header=not header_written)
                            val.to_pandas().to_csv(path_or_buf=save_file['valraw']['path'], mode='a', index=False, header=not header_written)
                            test.to_pandas().to_csv(path_or_buf=save_file['testraw']['path'], mode='a', index=False, header=not header_written)
                            if not header_written:
                                header_written = True
                        else:
                            trainraw.append(train)
                            valraw.append(val)
                            testraw.append(test)
                    if lowMemory:
                        if ahead > 1: fortran_order=False 
                        else: fortran_order=None 
                        # print(f'{train_idx = }')
                        # print(f'{val_idx = }')
                        # print(f'{x[:train_idx].shape = }')
                        # print(f'{y[:train_idx].shape = }')
                        # print(f'{x[train_idx:val_idx].shape = }')
                        # print(f'{y[train_idx:val_idx].shape = }')
                        # print(f'{x[val_idx:].shape = }')
                        # print(f'{y[val_idx:].shape = }')
                        # print(f'{y[:train_idx].shape = }')
                        save_file['xtrain']['writer'].append(x[:train_idx], fortran_order=fortran_order) 
                        save_file['ytrain']['writer'].append(y[:train_idx], fortran_order=fortran_order) 
                        save_file['xval']['writer'].append(x[train_idx:val_idx], fortran_order=fortran_order) 
                        save_file['yval']['writer'].append(y[train_idx:val_idx], fortran_order=fortran_order) 
                        save_file['xtest']['writer'].append(x[val_idx:], fortran_order=fortran_order) 
                        save_file['ytest']['writer'].append(y[val_idx:], fortran_order=fortran_order) 
                    else:
                        xtrain.extend(x[:train_idx]) 
                        ytrain.extend(y[:train_idx]) 
                        xval.extend(x[train_idx:val_idx]) 
                        yval.extend(y[train_idx:val_idx]) 
                        xtest.extend(x[val_idx:]) 
                        ytest.extend(y[val_idx:]) 
        else:
            d=df.sort(dateFeature)
            x, y, train_idx, val_idx, trainFeaturesAdded, _ = self._SplitCore(df=d, 
                                                                           dateFeature=dateFeature, 
                                                                           granularity=granularity,
                                                                           trainFeatures=trainFeatures, 
                                                                           targetFeatures=targetFeatures,
                                                                           lagWindow=lag_window, 
                                                                           lag=lag,
                                                                           ahead=ahead,
                                                                           splitRatio=splitRatio,
                                                                           segmentFeature=None,
                                                                           segmentFeatureEle=None)

            if any([x[:train_idx].size == 0, 
                    y[:train_idx].size == 0, 
                    (x[train_idx:val_idx].size == 0) & (splitRatio[1] != 0), 
                    (y[train_idx:val_idx].size == 0) & (splitRatio[1] != 0), 
                    (x[val_idx:].size == 0) & (splitRatio[2] != 0), 
                    (y[val_idx:].size == 0) & (splitRatio[2] != 0)]):
                print('No instance after splitting')
                exit()
            if saveRaw: 
                if lowMemory:
                    d[:train_idx].to_pandas().to_csv(path_or_buf=save_file['trainraw']['path'], mode='a', index=False, header=not header_written)
                    d[train_idx:val_idx].to_pandas().to_csv(path_or_buf=save_file['valraw']['path'], mode='a', index=False, header=not header_written)
                    d[val_idx:].to_pandas().to_csv(path_or_buf=save_file['testraw']['path'], mode='a', index=False, header=not header_written)
                    if not header_written:
                        header_written = True
                else:
                    trainraw = d[:train_idx]
                    valraw = d[train_idx:val_idx]
                    testraw = d[val_idx:]
            if lowMemory:
                np.save(file=save_file['xtrain']['path'], arr=x[:train_idx])
                np.save(file=save_file['ytrain']['path'], arr=y[:train_idx])
                np.save(file=save_file['xval']['path'], arr=x[train_idx:val_idx])
                np.save(file=save_file['yval']['path'], arr=y[train_idx:val_idx])
                np.save(file=save_file['xtest']['path'], arr=x[val_idx:])
                np.save(file=save_file['ytest']['path'], arr=y[val_idx:])
            else:
                xtrain = x[:train_idx]
                ytrain = y[:train_idx]
                xval = x[train_idx:val_idx]
                yval = y[train_idx:val_idx]
                xtest = x[val_idx:]
                ytest = y[val_idx:]
        
        for key in save_file.keys():
            if save_file[key]['writer'] is not None: save_file[key]['writer'].close()
        if lowMemory:
            xtrain = np.load(file=save_file['xtrain']['path'], mmap_mode='r+')
            ytrain = np.load(file=save_file['ytrain']['path'], mmap_mode='r+')
            xval = np.load(file=save_file['xval']['path'], mmap_mode='r')
            yval = np.load(file=save_file['yval']['path'], mmap_mode='r')
            xtest = np.load(file=save_file['xtest']['path'], mmap_mode='r')
            ytest = np.load(file=save_file['ytest']['path'], mmap_mode='r')

        if lowMemory and saveRaw:
            trainraw = self.ReadFileAddFetures(csvs=save_file['trainraw']['path'], 
                                               hasHeader=True,
                                               separator=',',
                                               tryParseDates=True,
                                               lowMemory=lowMemory, 
                                               fileNameAsFeature=None,
                                               storeValues=False)

            valraw = self.ReadFileAddFetures(csvs=save_file['valraw']['path'], 
                                               hasHeader=True,
                                               separator=',',
                                               tryParseDates=True,
                                               lowMemory=lowMemory, 
                                               fileNameAsFeature=None,
                                               storeValues=False)

            testraw = self.ReadFileAddFetures(csvs=save_file['testraw']['path'], 
                                               hasHeader=True,
                                               separator=',',
                                               tryParseDates=True,
                                               lowMemory=lowMemory, 
                                               fileNameAsFeature=None,
                                               storeValues=False)
        if not lowMemory and saveRaw and segmentFeature is not None:
            trainraw = pl.concat(trainraw)
            valraw = pl.concat(valraw)
            testraw = pl.concat(testraw)

        # np.save(Path(saveDir, 'temp.npy'), xtrain)
        # xtrain = np.load(Path(saveDir, 'temp.npy'), mmap_mode='r')
        xtrain = np.array(xtrain)
        # save_file['xtrain']['writer'].append(xtrain)
        # del(xtrain)
        # os.remove(Path(saveDir, 'temp.npy'))
        # print(f'{subname} - {xtrain.shape = }')
        ytrain = np.array(ytrain)
        xval = np.array(xval)
        yval = np.array(yval)
        if not self.skip_test:
            xtest = np.array(xtest)
            ytest = np.array(ytest)
        else:
            xtest = np.array(xtest[:10])
            ytest = np.array(ytest[:10])

        if storeValues:     
            self.xtrain = xtrain
            self.ytrain = ytrain
            self.xval = xval
            self.yval = yval
            self.xtest = xtest
            self.ytest = ytest

            if saveRaw:
                self.trainraw = trainraw
                self.valraw = valraw
                self.testraw = testraw
        else:
            # print(f'{xtrain.shape = }')
            return [xtrain, ytrain, xval, yval, xtest, ytest]
        