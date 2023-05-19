from utils.utils import Dataset

dataset = Dataset(dataset_name='api-2019',
                  csv_file='dataset/api-2019/api-2019.csv',
                  seq_len=100, 
                  fmt=['category', 'name', 'length', 'seq'])
dataset.initialize_word2index()
dataset.split_dataset(fold=5, ratio=0.8)
