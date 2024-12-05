from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt

class compute_metrics:
    def __init__(self, opt):
        if opt.dset_name in ['OCD_90_200_fMRI', 'PPMI']:
            self.avg = 'binary'
        else:
            self.avg = 'macro'
    def __call__(self, pred):
        labels = pred.label_ids  # 将ndarray转换为Python列表
        preds = pred.predictions.argmax(-1)  # 将ndarray转换为Python列表
        # precision, recall, f1, _ = precision_recall_fscore_support(
        #     labels, preds, average=self.avg)
        f1 = f1_score(labels, preds, average=None)
        recall = recall_score(labels, preds, average=None)
        precision = precision_score(labels, preds, average=None)
        # print(pred.label_ids.shape, pred.predictions.shape)
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


# def plot_acc(opt: database, model_name: str, num_epochs: int, accuracies: list):
#     '''
#     绘制准确率折线图.
#     '''
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(1, num_epochs+1), accuracies, "r.-")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title(db_cfg.name + " with " + model_name + " Validation Accuracy")
#     plt.show()