import os
import glob

class DatasetsLoader:
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.targets = args.targets
        self.train_name = self.targets[0] if len(self.targets) == 1 else "_".join(self.targets)
        self.dataset_path = os.path.join('./seqlen_{}_mels_{}'.format(args.seq_len, args.dims), self.dataset_name)
        self.pathList_dict = {
                              'train':[], 
                              'test':{}
                             }

        # list of directories ['~/Item_1/', '~/Item_2/', '~/Item_3/']
        self.dataDir_list = glob.glob(os.path.join(self.dataset_path, '*')) 
        
        self._on_memory()
        
    
    def _on_memory(self):
        def getPath_list(x, y):
            # Get list of data path in a directory
            path_list = glob.glob(os.path.join(f'{x}/{y}', '*'))

            return path_list
        
        trainDirs_list = []
        testDirs_list = []
        for dataDir_path in self.dataDir_list:
            data_name = os.path.basename(dataDir_path)
            
            if data_name in self.targets:
                trainDirs_list.append(dataDir_path)
            else:
                testDirs_list.append(dataDir_path)

        for dataDir_path in self.dataDir_list:
            data_name = os.path.basename(dataDir_path)
            
            # If the directory is target directory
            if data_name in self.targets: 
                self.pathList_dict['train'] += getPath_list(dataDir_path, 'train')
                self.pathList_dict['test'][data_name] = getPath_list(dataDir_path, 'test')
                
            else:
                testData_list = getPath_list(dataDir_path, 'train') + getPath_list(dataDir_path, 'test')
                self.pathList_dict['test'][data_name] = testData_list
                
                
