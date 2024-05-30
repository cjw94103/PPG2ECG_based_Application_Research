import json

def load_json_file(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        data = json.load(file)
    return data

def save_json(filename, data):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def load_jsonl_file(file_path, encoding='utf-8'):
    data = []
    with open(file_path, 'r', encoding=encoding) as file:
        for line in file:
            data.append(json.loads(line))
    return data

def save_jsonl(filename, data_list):
    with open(filename, 'w') as f:
        for data in data_list:
            json.dump(data, f)
            f.write('\n')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, max_len=-1):
        self.val = []
        self.count = []
        self.max_len = max_len
        self.avg = 0

    def update(self, val, n=1):
        self.val.append(val * n)
        self.count.append(n)
        if self.max_len > 0 and len(self.val) > self.max_len:
            self.val = self.val[-self.max_len:]
            self.count = self.count[-self.max_len:]
        self.avg = sum(self.val) / sum(self.count)