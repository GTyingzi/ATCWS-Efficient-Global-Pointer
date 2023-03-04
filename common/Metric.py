import torch
import numpy as np
from itertools import chain

class MetricsCalculator(object):
    def __init__(self):
        super().__init__()
        self.R_list = []
        self.T_list = []
        self.R_label = {}
        self.T_label = {}

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        X,Y,Z = 1e-10,1e-10,1e-10
        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall

    def get_evaluate_temp2_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
            if l in self.R_label.keys():
                self.R_label[l].append((b, l, start, end))
            else:
                self.R_label[l] = [(b, l, start, end)]
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))
            if l in self.T_label.keys():
                self.T_label[l].append((b, l, start, end))
            else:
                self.T_label[l] = [(b, l, start, end)]

        self.R_list.append(pred)
        self.T_list.append(true)

    def get_evaluate_2_fpr(self):
        # 处理嵌套list
        R = list(chain(*self.R_list))
        T = list(chain(*self.T_list))

        R = set(R)
        T = set(T)

        X = len(R & T)
        Y = len(R)
        Z = len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall

    def get_evaluate_2_label_fpr(self,labels):
        evaluate= {}
        avg_f1,avg_p,avg_r = 0,0,0
        for label,id in labels.items():
            R = set(self.R_label[id])
            T = set(self.T_label[id])

            X = len(R & T)
            Y = len(R)
            Z = len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            evaluate[label] = {"f1":f1,"precision":precision,"recall":recall}

            avg_f1 += f1
            avg_p += precision
            avg_r += recall

        avg_evaluate = {"avg_f1":avg_f1/len(labels),"avg_p":avg_p/len(labels),"avg_r":avg_r/len(labels)}

        return evaluate,avg_evaluate



