import os
import json
import polars as pl 
from train import run
from copy import deepcopy 
from itertools import combinations

template = {
	'path': [],
	'target': None,
	'date': None,
	'features': None,
	'delimiter': ",",
	'segmentFeature': None,
	'timeFormat': None,
	'granularity': None,
	'fileNameAsFeature': None,
	'hasHeader': True,
	'stripTime': None
}

datasets = [
	{
		'path': r'.\data\Salinity\processed\by_station-mekong',
		'target': 'average',
		'date': 'date',
		'features': ['station'],
		'segmentFeature': 'station',
		'granularity': 1440,
		'fileNameAsFeature': 'station',
	},{
		'path': r'.\\data\\CryptoDataDownload_Com\\processed\\CryptoDataDownload_Com-Binance-Day.csv',
		'date': 'Date',
		'features': ['Symbol', 'Open', 'High', 'Low', 'VolumeSymbol1', 'VolumeSymbol2', 'TradeCount'],
		'target': 'Close',
		'segmentFeature': 'Symbol', 
		'granularity': 1440,
	},{
		'path': r'.\data\WeatherStationBeutenberg\processed\WeatherStationBeutenberg.csv',
		'date': 'Date Time',
		'features': ['p (mbar)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max_wv_m_per_s', 'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR_W_per_m2', 'PAR_umol_per_m2_s', 'max. PAR_umol_per_m2_s', 'Tlog (degC)', 'CO2 (ppm)'],
		'target': 'T (degC)',
		'granularity': 10,
	},{
		'path': r'.\data\WeatherStationSaaleaue\processed\WeatherStationSaaleaue.csv',
		'date': 'Date Time',
		'features': ['p (mbar)', 'rh (%)', 'sh (g/kg)', 'Tpot (K)', 'Tdew (degC)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'wd (deg)', 'rain (mm)', 'SWDR (W/m**2)', 'SDUR (s)', 'TRAD (degC)', 'Rn (W/m**2)', 'ST002 (degC)', 'ST004 (degC)', 'ST008 (degC)', 'ST016 (degC)', 'ST032 (degC)', 'ST064 (degC)', 'ST128 (degC)', 'SM008 (%)', 'SM016 (%)', 'SM032 (%)', 'SM064 (%)', 'SM128 (%)'],
		'target': 'T (degC)',
		'granularity': 10,
	},{
		'path': r'.\data\DailyTemperaturesJena\processed\Daily_Temperatures_Jena.csv',
		'date': 'Date',
		'features': ['Tmin', 'Tmax'],
		'target': 'Tmean',
		'granularity': 1440,
	},{
		'path': r'.\data\ElectricityTransformerDataset\raw\ETT-small\ETTh1.csv',
		'date': 'date',
		'features': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'],
		'target': 'OT',
		'granularity': 60,
	},{
		'path': r'.\data\ElectricityTransformerDataset\raw\ETT-small\ETTh2.csv',
		'date': 'date',
		'features': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'],
		'target': 'OT',
		'granularity': 60,
	},{
		'path': r'.\data\ElectricityTransformerDataset\raw\ETT-small\ETTm1.csv',
		'date': 'date',
		'features': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'],
		'target': 'OT',
		'granularity': 15,
	},{
		'path': r'.\data\ElectricityTransformerDataset\raw\ETT-small\ETTm2.csv',
		'date': 'date',
		'features': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'],
		'target': 'OT',
		'granularity': 15,
	},{
		'path': r'.\data\IndividualHouseholdElectricPowerConsumption\processed\household_power_consumption.csv',
		'date': 'Date',
		'features': ['Global_reactive_power', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Global_active_power'],
		'target': 'Voltage',
		'granularity': 1,
	}
]

if __name__ == '__main__':
	path = 'cases-paper.xlsx'
	df = pl.read_excel(path)
	data = df.to_dicts()
	project = 'paper'

	for datum in data:
		if datum["Case"] not in [6, 7, 13, 14, 20, 21, 27, 28]: continue
		# if datum["Case"] < 22: continue
		if project is None:
			project = datum['project']

		weights = {}
		if datum['weight'] != 'None':
			for weight in os.listdir(os.path.join(project, datum['weight'], 'weights')):
				# print(f"{os.path.join('runs', datum['weight'], 'weights', weight) = }")
				# if 'LTSF' in weight:
				# 	weights[f'{weight}_weight'] = os.path.join(project, datum['weight'], 'weights', weight, 'best', f'{weight}_best.index')
				weights[f'{weight}_weight'] = os.path.join(project, datum['weight'], 'weights', weight, 'best', f'{weight}_best.index')
		if datum['normalization'] == 'None':
			datum['normalization'] = None
		configs = []
		# print(datum['DataType'])
		if datum['DataType'] == 'Multivariate':
			column_dict = {column: index for index, column in enumerate(datum.keys())}
			input_features = -1
			for dataset in list(datum.keys())[column_dict['Salinity']:]:
				if not datum[dataset]: continue
				for subdataset in datasets:
					if dataset in subdataset['path']:
						dataset = subdataset
						if input_features == -1:
							input_features = len(dataset['features'])
							print(f'{input_features = }')
						break
				if 'target' in dataset:
					# print(dataset['features'])
					assert len(dataset['features']) >= input_features, 'Number of features of train/val dataset less than test set => cannot permutate'
					for comb in list(combinations(dataset['features'], input_features)):
						temp = deepcopy(template)
						temp['path'] = [dataset['path']]
						temp['date'] = dataset['date']
						temp['features'] = list(comb)
						temp['target'] = dataset['target']
						temp['granularity'] = dataset['granularity']
						if 'segmentFeature' in dataset: temp['segmentFeature'] = dataset['segmentFeature']
						if 'fileNameAsFeature' in dataset: temp['fileNameAsFeature'] = dataset['fileNameAsFeature']
						configs.append(temp)
				else:
					assert len(dataset['features']) >= input_features + 1, 'Number of features of train/val dataset less than test set => cannot permutate'
					for comb in list(combinations(dataset['features'], input_features + 1)):
						temp = deepcopy(template)
						temp['path'] = dataset['path']
						temp['date'] = dataset['date']
						temp['features'] = list(comb[1:])
						temp['target'] = comb[0]
						temp['granularity'] = dataset['granularity']
						if 'segmentFeature' in dataset: temp['segmentFeature'] = dataset['segmentFeature']
						if 'fileNameAsFeature' in dataset: temp['fileNameAsFeature'] = dataset['fileNameAsFeature']
						configs.append(temp)
		elif datum['DataType'] == 'Univariate':
			input_features = 1
			column_dict = {column: index for index, column in enumerate(datum.keys())}
			for dataset in list(datum.keys())[column_dict['Salinity']:]:
				if not datum[dataset]: continue
				for subdataset in datasets:
					if dataset in subdataset['path']:
						dataset = subdataset
						break
				
				features = [*dataset['features']]
				if 'target' in dataset: features.append(dataset['target'])
				if 'segmentFeature' in dataset:
					features.remove(dataset['segmentFeature'])
				
				for feature in features:
					temp = deepcopy(template)
					temp['path'] = [dataset['path']]
					temp['date'] = dataset['date']
					temp['features'] = []
					temp['target'] = [feature]
					temp['granularity'] = dataset['granularity']
					if 'segmentFeature' in dataset: temp['segmentFeature'] = dataset['segmentFeature']
					if 'fileNameAsFeature' in dataset: temp['fileNameAsFeature'] = dataset['fileNameAsFeature']
					configs.append(temp)
		print(json.dumps({'dataConfigs':configs,
			'low_memory':datum['low_memory'],
			'batchsz':datum['batchsz'],
			'ensemble':datum['ensemble'],
			'normalization':datum['normalization'],
			'lag':datum['lag'],
			'ahead':datum['ahead'],
			'offset':datum['offset'],
			'lr':datum['lr'],
			'name':f'case{datum["Case"]}'}, indent=4))

		print(weights)
		print(datum)
		run(
			dataConfigs=configs,
			low_memory=datum['low_memory'],
			batchsz=datum['batchsz'],
			ensemble=datum['ensemble'],
			LTSF=datum['LTSF'],
			ResLSTM__Tensorflow=datum['ResLSTM'],
			LTSF_NDLinearTime2VecRevIN__Tensorflow=datum['NDLinearTime2VecRevIN'],
			normalization=datum['normalization'],
			lag=datum['lag'],
			ahead=datum['ahead'],
			offset=datum['offset'],
			lr=datum['lr'],
			name=f'case{datum["Case"]}',
			epochs=datum['epochs'],
			patience=datum['patience'],
			project=project,
			**weights
		)
