import torch
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
import os
import config
import jieba
import logging
jieba.setLogLevel(logging.INFO)

def load_ent2id(ent2id_path):
    return json.load(open(ent2id_path, encoding="utf-8"))

eval_config = config.eval_config
config = config.train_config
hyper_parameters = config["hyper_parameters"]


ent2id_path = os.path.join(config["data_home"], config["exp_name"], config["ent2id"])
ent2id = load_ent2id(ent2id_path)

tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=True, do_lower_case=False)


class EntDataset(Dataset):
    def __init__(self, data, tokenizer, istrain=True,istoken=False):
        self.data = data
        self.tokenizer = tokenizer
        self.istrain = istrain
        self.istoken = istoken # 是否引入中文分词

    def __len__(self):
        return len(self.data)

    # 引入分词信息
    def get_seg_lables(self,sentence,start_mapping,end_mapping,max_len=128):
        jieba_list = list(jieba.cut(sentence))
        text_len = 0
        word_list = []
        for j in jieba_list:
            start,end = text_len,text_len + len(j) - 1
            if start in start_mapping and end in end_mapping:
                start_index = start_mapping[start]
                end_index = end_mapping[end]
                word_list.append((start_index,end_index))
            text_len += len(j)

        seq_len = max_len
        seg_labels = torch.zeros((seq_len, seq_len))
        cut_head = [i[0] for i in word_list]
        cut_tail = [i[1] for i in word_list]
        np.zeros(seq_len)
        tail = torch.zeros(seq_len)
        tail[torch.tensor(cut_tail)] = 1
        seg_labels[torch.tensor(cut_head)] = tail
        seg_labels = torch.triu(seg_labels, diagonal=0)
        return seg_labels.numpy()

    def encoder(self, item):
        if self.istrain:
            text = item[0]
            token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True, max_length=hyper_parameters["max_seq_len"], truncation=True)["offset_mapping"]
            start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
            end_mapping = {j[-1] - 1: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
            #将raw_text的下标 与 token的start和end下标对应
            encoder_txt = self.tokenizer.encode_plus(text, max_length=hyper_parameters["max_seq_len"], truncation=True)
            input_ids = encoder_txt["input_ids"]
            token_type_ids = encoder_txt["token_type_ids"]
            attention_mask = encoder_txt["attention_mask"]

            return text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask
        else:
            #TODO 测试
            pass

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def collate(self, examples):
        batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids = [], [], [], []
        raw_text_list,seg_labels_list = [],[]
        for item in examples:
            raw_text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask = self.encoder(item)

            seg_labels = self.get_seg_lables(raw_text,start_mapping,end_mapping,hyper_parameters["max_seq_len"])
            labels = np.zeros((len(ent2id), hyper_parameters["max_seq_len"], hyper_parameters["max_seq_len"]))
            for start, end, label in item[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[label, start, end] = 1
            raw_text_list.append(raw_text)

            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])
            seg_labels_list.append(seg_labels[:len(input_ids),:len(input_ids)])

        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segmentids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attentionmask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()
        batch_seg_labels = torch.tensor(self.sequence_padding(seg_labels_list,seq_dims=2)).long()

        if self.istoken:
            return raw_text_list, batch_seg_labels,batch_inputids, batch_attentionmask, batch_segmentids, batch_labels
        else:
            return raw_text_list, batch_inputids, batch_attentionmask, batch_segmentids, batch_labels

    def __getitem__(self, index):
        item = self.data[index]
        return item

def data_generator(data_type='train',exp_name='CMeEE'):
    if exp_name == "CMeEE":
        data_path = os.path.join(config["data_home"], config["exp_name"], config["train_data"])
        data = get_CMeEE_data(data_path)
    elif exp_name == "CCKS2019":
        train_data_path = os.path.join(config["data_home"], config["exp_name"], config["train_data"])
        test_data_path = os.path.join(config["data_home"], config["exp_name"], eval_config["test_data"])
        data = get_CCKS2019_data(train_data_path,test_data_path)

    if data_type == 'train':
        train_data = EntDataset(data['train_data'],tokenizer,istoken=True)
        train_data_loader = DataLoader(train_data,
                                    batch_size=hyper_parameters["batch_size"],
                                    shuffle=True,
                                    num_workers=config["num_workers"],
                                    collate_fn=train_data.collate
                                       )
        return train_data_loader
    elif data_type == 'val':
        val_data = EntDataset(data['val_data'],tokenizer)
        val_data_loader = DataLoader(val_data,
                                     batch_size=hyper_parameters["batch_size"],
                                     shuffle=False,
                                     num_workers=config["num_workers"],
                                     collate_fn=val_data.collate)
        return val_data_loader
    elif data_type == 'test':
        test_data = EntDataset(data['test_data'],tokenizer)
        test_data_loader = DataLoader(test_data,
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=False,
                                      num_workers=config["num_workers"],
                                      collate_fn=test_data.collate)
        return test_data_loader

def get_CMeEE_data(data_path):
    def load_CMeEE_data(path):
        datas = []
        for data in json.load(open(path)):
            datas.append([data['text']])
            for e in data['entities']:
                start, end, label = e['start_idx'], e['end_idx'], e['type']
                if start <= end:
                    datas[-1].append((start, end, ent2id[label]))
        return datas

    data = load_CMeEE_data(data_path)
    train_data,test_data = train_test_split(data,test_size=0.3)
    val_data,test_data = train_test_split(test_data,test_size=0.5)
    all_data = {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data
    }
    return all_data

def get_CCKS2019_data(train_path,test_path):
    def load_CCKS2019_data(path):
        datas = []
        with open(path, encoding="utf-8-sig") as f:
            for data in f:
                data = json.loads(data)
                datas.append([data["originalText"]])
                for e in data["entities"]:
                    start,end,label = e["start_pos"],e["end_pos"],e["label_type"]
                    if start <= end:
                        datas[-1].append((start, end, ent2id[label]))
        return datas

    train_data = load_CCKS2019_data(train_path)
    test_data = load_CCKS2019_data(test_path)
    train_data,val_data = train_test_split(train_data,test_size=0.2)
    all_data = {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data
    }
    return all_data

if __name__ == '__main__':
    train_dataloader = data_generator(data_type="train",exp_name='CCKS2019')
