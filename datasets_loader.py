import os
import glob


class DatasetsLoader:
    datasets_2018 = {
        'AT'   : {'name':'AT2-IN88-SINK'},
        'M1'   : {'name':'M-3708'},
        'M2'   : {'name':'M-4478'},
        'NA1'  : {'name':'NA-9289-MAIN'},
        'NA2'  : {'name':'NA-9473'},
        'ST'   : {'name':'ST-4214-GE'}
    }
    datasets_2019_1 = {
        'CLR'  : {'name':'CLR-085'},
        'MG'   : {'name':'MG-A121H'},
        'ST1'  : {'name':'ST-2624'},
        'ST2'  : {'name':'ST-2744-GE'},
        'ST3'  : {'name':'ST-3428'},
        'TSIO' : {'name':'TSIO-2002'},
        'NW'   : {'name':'NW'}
    }
    datasets_2019_2 = {
        'CLR-2'  : {'name':'CLR-085'},
        'MA'   : {'name':'MA-A097E'},
        'MG-2'  : {'name':'MG-A200B'},
        'NA-2'  : {'name':'NA-9285'},
    }
    
    def __init__(self, targets, data_type, augment=False):
        self.targets = targets
        self.train_name = self.targets[0] if len(self.targets) == 1 else "_".join(self.targets)
        self.augment = augment
        self.pathList_dict = {'train':{}, 'test':{}}
        
        for target in self.targets:
            if target in DatasetsLoader.datasets_2018:
                self.datasets = DatasetsLoader.datasets_2018
                break
            
            if target in DatasetsLoader.datasets_2019_1:
                self.datasets = DatasetsLoader.datasets_2019_1
                break
                
            if target in DatasetsLoader.datasets_2019_2:
                self.datasets = DatasetsLoader.datasets_2019_2
                break
                
        for key, value in self.datasets.items():
            name = value['name']
            path = os.path.join(f'./{data_type}', name)
            self.datasets[key]['path'] = path
        
        self._on_memory()
        
    def _on_memory(self):
        getPath_list = lambda x, y: glob.glob(os.path.join(f'{x}/{y}', '*'))
        
        targetTrain_list = []
        targetTest_list = []
        for target, dataset in self.datasets.items():
            dataset_path = dataset['path']

            if target in self.targets:
                dataPath_list = getPath_list(dataset_path, 'train')
                dataPath_list = dataPath_list + getPath_list(dataset_path, 'train_shifted') if self.augment==True else dataPath_list
                
                targetTrain_list += dataPath_list
                
                dataPath_list = getPath_list(dataset_path, 'test')
                dataPath_list = dataPath_list + getPath_list(dataset_path, 'test_shifted') if self.augment==True else dataPath_list
                
                targetTest_list += dataPath_list
            else:
                dataPath_list = getPath_list(dataset_path, 'test')
                dataPath_list = dataPath_list + getPath_list(dataset_path, 'test_shifted') if self.augment==True else dataPath_list
                
                self.pathList_dict['test'][target] = dataPath_list
                
        self.pathList_dict['train'][self.train_name] = targetTrain_list
        self.pathList_dict['test'][self.train_name] = targetTest_list