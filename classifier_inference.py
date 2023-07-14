import glob

import torch
import os
from tqdm import tqdm

from classifier import Classifier
from dataloader import CelebA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

#Doing inference  function, return accuracy, precision, recall, f1-score, confusion matrix in the txt file.
def inference(classifier, dataloader,save_name):
    print("start inference")
    result = open(save_name+".txt",mode="a",encoding="utf-8")
    with open("classifier_feature_record.txt", "w") as f:
        for i, (x, label, im_name) in tqdm(enumerate(dataloader)):
            x = x.cuda()
            with torch.no_grad():
                y, mu, log_var = classifier(x)

            BS = x.shape[0]
            for j in range(BS):
                f.write(f"{im_name[j]}  {label[j].item()}  {y[j].item()}  {mu[j].cpu().numpy().tolist()}\n")

    with open("classifier_feature_record.txt", "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    y_label = [int(line.split("  ")[1]) for line in lines]
    y_pred = [int(float(line.split("  ")[2]) > 0) for line in lines]
    print("=====================================================save name: ",save_name,"\n",file = result)
    print("Accuracy: \n", accuracy_score(y_label, y_pred), "\n", file = result)
    print("Confusion Matrix: \n", confusion_matrix(y_label, y_pred),"\n", file = result)
    print("Classification Report: \n", classification_report(y_label, y_pred),"\n", file = result)
    result.close()


#if want to inference separately, need to modify the following lines
# change load_pickle to the pickle you want to load
# change dataset to the dataset you want to inference

#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# classifier = Classifier().to(DEVICE)
# # load_pickle = "pickles/classifier_aaa_epoch_99_0.0029717.pickle"
# load_pickle = glob.glob("pickles/classifier_aaa_epoch_99_*.pickle")[-1]
# if os.path.exists(load_pickle):
#     print("load pickle: " +load_pickle +"\n")
#     with open(load_pickle, "rb") as f:
#         state_dict = torch.load(f, map_location=DEVICE)
#     classifier.load_state_dict(state_dict)
#
# dataset = CelebA("/home/robotlab/Desktop/img_align_celeba_png", usage="test", shuffle="Yes")
#
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

# with open("classifier_feature_record.txt", "w") as f:
#     for i, (x, label, im_name) in tqdm(enumerate(dataloader)):
#         x = x.cuda()
#         with torch.no_grad():
#             y, mu, log_var = classifier(x)
#
#         BS = x.shape[0]
#         for j in range(BS):
#             f.write(f"{im_name[j]}  {label[j].item()}  {y[j].item()}  {mu[j].cpu().numpy().tolist()}\n")
#
# with open("classifier_feature_record.txt", "r") as f:
#     lines = f.readlines()
# lines = [line.strip() for line in lines]
# y_label = [int(line.split("  ")[1]) for line in lines]
# y_pred = [int(float(line.split("  ")[2]) > 0) for line in lines]
#
# print("Accuracy: \n", accuracy_score(y_label, y_pred))
# print("Confusion Matrix: \n", confusion_matrix(y_label, y_pred))
# print("Classification Report: \n", classification_report(y_label, y_pred))