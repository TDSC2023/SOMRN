import math, os, csv, json, random, time
import torch
import pickle
import numpy as np
from collections import defaultdict
from sklearn.utils import compute_class_weight

class Dataset:
    def __init__(self, dataset_name, csv_file, seq_len, fmt: list):
        self.dataset_name = dataset_name
        self.vector_type = None
        self.mode = None
        self.dataset_info = None
        self.dataset_index = None
        self.word2index = None
        self.index2vec = None
        self.category = None
        self.zero_padding = None
        self.seq_len = seq_len
        self.fmt = fmt
        self.text_feature_extraction_data_file = None

        with open(csv_file, 'r', encoding='utf-8') as fp:
            self.data = []
            s = 0
            for row in csv.reader(fp):
                self.data.append(list(filter(lambda x: len(x) > 0, row)))
            self.data = self.data[:100]

    def initialize_word2index(self, overwrite=False):
        path = './dataset/{}/word2index.json'.format(self.dataset_name)
        if overwrite or not os.path.exists(path): 
            opcodes = set()
            for record in self.data:  
                for opcode in record[self.fmt.index('seq'):]:
                    opcodes.add(opcode) 
            self.word2index = {opcode: index for index, opcode in enumerate(sorted(opcodes))}
            json.dump(
                obj=self.word2index,
                fp=open(path, 'w', encoding='utf8'),
                indent=4
            )
        else:  
            self.word2index = json.load(open(path, 'r', encoding='utf8'))
        self.split_dataset(5, 0.8, overwrite)

    def split_dataset(self, fold, ratio, overwrite=False): 
        if overwrite or not os.path.exists('./dataset/{}/dataset_info.json'.format(self.dataset_name)):
            categories = defaultdict(list)
            for index, record in enumerate(self.data):
                categories[record[self.fmt.index('category')]].append(index)
            dataset_info = { 
                'train_set': [],
                'test_set': [],
                'all': []
            }
            for k in range(fold):
                dataset_info['fold_%d' % k] = []
            for category in categories.values():
                random.shuffle(category)
                trainset_num = int(ratio * len(category))
                dataset_info['train_set'].extend(category[:trainset_num])
                dataset_info['test_set'].extend(category[trainset_num:])
                dataset_info['all'].extend(category)
                avg_list = self.avg_split_list(category[:trainset_num], fold)
                for k in range(0, fold):
                    dataset_info['fold_%d' % k].extend(avg_list[k])
            self.dataset_info = dataset_info
            self.category = len(categories)
            json.dump(
                obj=dataset_info,
                fp=open('./dataset/{}/dataset_info.json'.format(self.dataset_name), 'w', encoding='utf8'),
                indent=4
            )
            json.dump(
                obj=categories,
                fp=open('./dataset/{}/categories.json'.format(self.dataset_name), 'w', encoding='utf8'),
                indent=4
            )

    def set_category(self, category):
        self.category = category

    def set_vector_type(self, vector_type, index2vec=None):
        self.vector_type = vector_type
        if vector_type == 'one-hot':
            n_apis = len(self.word2index) 
            self.zero_padding = [0 for _ in range(n_apis)]
            self.index2vec = []
            for i in range(n_apis):
                one_hot_vec = [0 for _ in range(n_apis)]
                one_hot_vec[i] = 1
                self.index2vec.append(one_hot_vec)
        elif vector_type == 'reduced':
            self.index2vec = pickle.load(open(index2vec, 'rb'))
            self.zero_padding = [0 for _ in range(len(self.index2vec[0]))]
        else:
            raise Exception('undefined vector type')

    def set_mode(self, mode, validate=False, k=None):
        self.mode = mode
        if self.dataset_info is None:
            self.dataset_info = json.load(open('./dataset/{}/dataset_info.json'.format(self.dataset_name), 'r', encoding='utf8'))
        if self.mode in ('train_set', 'test_set', 'all'):
            self.dataset_index = self.dataset_info[self.mode]
        elif self.mode == 'cross_validation':
            if validate:
                self.dataset_index = self.dataset_info['fold_%d' % k]
            else:
                fold = len(self.dataset_info) - 3
                cv_set = []
                for i in range(fold):
                    if i != k:
                        cv_set.extend(self.dataset_info['fold_%d' % i])
                self.dataset_index = cv_set
        else:
            raise Exception('undefined mode')

    def compute_class_weight(self):
        y = []
        for record in self.data:
            y.append(int(record[self.fmt.index('category')]) - 1)
        return compute_class_weight('balanced', classes=np.unique(y), y=y)

    @staticmethod
    def avg_split_list(obj, fold):
        points = np.linspace(0, len(obj), fold + 1, endpoint=True, dtype=int)   # 0 1 2 3 4 5 共6个点
        res = []
        for i in range(fold):
            res.append(list(obj[points[i]:points[i + 1]]))
        return res


    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, item):
        record = self.data[self.dataset_index[item]]
        seq = record[self.fmt.index('seq'):] 
        seq = map(lambda x: self.word2index[x], seq)  
        seq = map(lambda x: self.index2vec[x], seq)  
        label = int(record[self.fmt.index('category')]) - 1  
        seq = list(seq)[:self.seq_len]
        if len(seq) < self.seq_len:
            seq.extend([self.zero_padding for _ in range(self.seq_len - len(seq))])
        seq = torch.FloatTensor(seq)
        label = torch.LongTensor([label, ])
        return seq, label

    def load_text_feature_extraction_data(self):
        dataset = []
        cate_dict = {}
        for record in self.data: 
            cate = str(int(record[self.fmt.index('category')])-1)
            if cate not in cate_dict:
                cate_dict[cate] = 1
            else:
                cate_dict[cate] += 1
            dataset.append([cate, record[self.fmt.index('seq'):]])
        return dataset, cate_dict


class Reward:
    def __init__(self):
        self.neg_log_prob = torch.nn.CrossEntropyLoss()

    def __call__(self, predict, label, bias):
        return -self.neg_log_prob(predict, label) + bias


class W2VWrapper: 
    def __init__(self, dataset, radius, num=3):
        self.dataset = dataset
        self.radius = radius
        self.num = num

    def __len__(self):
        return len(self.dataset) * self.num

    def __getitem__(self, item):
        index = item // self.num
        offset = item % self.num
        seq, _ = self.dataset[index] 
        rows, cols = seq.shape  
        chunks = torch.chunk(seq[:self.num * (2 * self.radius + 1)], chunks=self.num, dim=0)  
        cache = []
        for i in range(self.num):
            seq = chunks[i]  
            corpus = torch.zeros(2 * self.radius, cols)
            corpus[:self.radius] = seq[:self.radius] 
            corpus[self.radius:] = seq[self.radius + 1: 2 * self.radius + 1]
            label = torch.argmax(seq[self.radius])  
            target = seq[self.radius]
            cache.append((corpus, label, target))
        return cache[offset]


class ParameterContainer:
    def __init__(self, **kwargs):
        self.parameters = kwargs

    def __getattr__(self, item):
        return self.parameters[item]


class Recorder:
    def __init__(self, path, hyper_parameters, arg_dicts, T=None, date_fmt="%Y-%m-%d %H:%M:%S"):
        self.path = path
        self.arg_dicts = arg_dicts
        self.date_fmt = date_fmt
        self.start_time = time.time()
        self.end_time = None
        self.T = T 
        self.BC = 0 
        self.record = {
            'hyper_parameters': hyper_parameters,
        }

    def write(self, *args, **kwargs):
        temp_dict = self.record
        for idx, arg in enumerate(args):
            key = '{}_{}'.format(self.arg_dicts[idx]['name'], arg)
            if key not in temp_dict:
                temp_dict[key] = {}
            temp_dict = temp_dict[key]
        for key, value in kwargs.items():
            if key not in temp_dict:
                temp_dict[key] = [value]
            else:
                temp_dict[key].append(value)
        self.BC += 1
        msg = ""
        for idx, arg in enumerate(args):
            msg += "({0}:{1}/{2})".format(self.arg_dicts[idx]['name'], arg, self.arg_dicts[idx]['max_value'] - 1)
        msg += "  " if self.T is None else "(remaining time:{:.2f}s)  ".format(
            (self.T / self.BC - 1) * (time.time() - self.start_time)
        )
        for key, value in kwargs.items():
            msg += "{}={:.6f}  ".format(key, value)
        now = time.strftime("%H:%M:%S")
        print("\r[{}]{}".format(now, msg), end='')

    def reduce(self, *args, func):
        temp_dict = self.record
        for idx, arg in enumerate(args):
            key = '{}_{}'.format(self.arg_dicts[idx]['name'], arg)
            temp_dict = temp_dict[key]
        for key, value in temp_dict.items():
            temp_dict[key] = func(value)

    def strftime(self, secs): 
        return time.strftime(self.date_fmt, time.localtime(secs))

    def save(self):
        self.end_time = time.time()
        self.record['start_time'] = self.strftime(self.start_time)
        self.record['end_time'] = self.strftime(self.end_time)
        self.record['time_consumption(secs)'] = self.end_time - self.start_time
        with open(self.path, mode='w') as fp:
            json.dump(
                obj=self.record,
                fp=fp,
                indent=4
            )

    def save_numpy(self, directory, filename, numpy_obj):
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(directory + '/{}.npy'.format(filename), numpy_obj)


class FeatureExtract:

    def __init__(self, Date):
        self.Date = Date
        self.dataset, self.cate_dict = self.Date.load_text_feature_extraction_data()  
        self.cate_nums = len(self.cate_dict)

    def collect_dfdict(self):
        worddf_dict = {}
        for data in self.dataset:
            category = data[0]
            for word in set(data[1]):
                if word not in worddf_dict:
                    worddf_dict[word] = category
                else:
                    worddf_dict[word] += '@' + category

        for word, word_category in worddf_dict.items():
            cate_dict = {}
            for cate in word_category.split('@'):
                if cate not in cate_dict:
                    cate_dict[cate] = 1
                else:
                    cate_dict[cate] += 1
            worddf_dict[word] = cate_dict
        return worddf_dict

    def DF(self, feature_num):
        df_dict = {}
        for data in self.dataset:
            for word in set(data[1]):
                if word not in df_dict:
                    df_dict[word] = 1
                else:
                    df_dict[word] += 1
        df_dict = sorted(df_dict.items(), key=lambda asd:asd[1], reverse=True)[:feature_num]
        features = [item[0] for item in df_dict]
        return features

    def CHI(self, feature_num):
        worddf_dict = self.collect_dfdict()
        N = sum(self.cate_dict.values())
        chi_dict = {}
        for word, word_cate in worddf_dict.items():
            data = {}
            for cate in range(self.cate_nums):
                cate = str(cate)
                A = word_cate.get(cate, 0)
                B = sum([word_cate[key] for key in word_cate.keys() if key != cate])
                C = self.cate_dict[str(cate)] - A
                D = N - self.cate_dict[str(cate)] - B
                chi_score = (N*(A*D - B*C)**2)/((A+C)*(A+B)*(B+D)*(B+C))
                data[cate] = chi_score
            chi_dict[word] = data

        features = self.select_best(feature_num, chi_dict)
        return features

    def IG(self, feature_num):
        worddf_dict = self.collect_dfdict()
        N = sum(self.cate_dict.values())
        ig_dict = {}
        for word, word_cate in worddf_dict.items():
            HC = 0.0 
            HTC = 0.0 
            HT_C = 0.0
            for cate in range(self.cate_nums):
                cate = str(cate)
                N1 = self.cate_dict[cate]
                hc = -(N1/N) * math.log(N1/N)
                A = word_cate.get(cate, 0)
                B = sum([word_cate[key] for key in word_cate.keys() if key != cate])
                C = self.cate_dict[str(cate)] - A
                D = N - self.cate_dict[str(cate)] - B
                p_t = (A + B) / N
                p_not_t = (C + D)/ N
                p_t_c = (A + 1)/ (A + B + self.cate_nums)
                p_t_not_c = (C + 1)/ (C + D + self.cate_nums)
                h_t_ci = p_t * p_t_c * math.log(p_t_c)
                h_t_not_ci = p_not_t * p_t_not_c * math.log(p_t_not_c)
                HC += hc
                HTC += h_t_ci
                HT_C += h_t_not_ci
            ig_score = HC + HTC + HT_C
            ig_dict[word] = ig_score

        ig_dict = sorted(ig_dict.items(), key=lambda asd:asd[1], reverse=True)[:feature_num]
        features = [item[0] for item in ig_dict]
        return features

    def MI(self, feature_num):
        worddf_dict = self.collect_dfdict()
        N = sum(self.cate_dict.values())
        mi_dict = {}
        for word, word_cate in worddf_dict.items():
            data = {}
            for cate in range(self.cate_nums):
                cate = str(cate)
                A = word_cate.get(cate, 0)
                B = sum([word_cate[key] for key in word_cate.keys() if key != cate])
                C = self.cate_dict[str(cate)] - A
                D = N - self.cate_dict[str(cate)] - B
                p_t_c = ( A + 1)/(A + B + self.cate_nums)
                p_c = (A + C) / N
                p_t = (A + B) / N
                mi_score = p_t_c * math.log(p_t_c / (p_c * p_t))
                data[cate] = mi_score
            mi_dict[word] = data

        features = self.select_best(feature_num, mi_dict)
        return features

    def select_best(self, feature_num, word_dict):
        cate_worddict = {}
        features = []
        for word, scores in word_dict.items():
            for cate, word_score in scores.items():
                if cate not in cate_worddict:
                    cate_worddict[cate] = {}
                else:
                    cate_worddict[cate][word] = word_score
        top_num = int(feature_num/self.cate_nums) + 100

        for cate, words in cate_worddict.items():
            words = sorted(words.items(), key=lambda asd: asd[1], reverse=True)[:top_num]
            top_words = [item[0] for item in words]
            features += top_words

        return list(set(features[:feature_num]))

# dataer = FeatureExtract()
# features = dataer.DF(5000)
#
# f = open('data/df.txt', 'w+')
# f.write('\n'.join(features))
# f.close()

if __name__ == '__main__':
    dataset = Dataset('../dataset/x86_samples.csv', 100, fmt=['category', 'name', 'length', 'seq'])
    dataer = FeatureExtract(dataset)

    features = dataer.IG(1000)
    print(features[:20])
    print(features[-20:])
