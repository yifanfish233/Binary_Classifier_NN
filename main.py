import glob
import sys
import torch
import os


from train_vlisulization_visdom import Visualizer
from dataloader import CelebA
from classifier import Classifier
from train_classifier import train
from classifier_inference import inference

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(len(sys.argv)>2):
        retrain_load = sys.argv[1]
    else:
        retrain_load = "None"
    classifier = Classifier().to(DEVICE)

    #if need retrain, uncomment following line
    load_pickle = retrain_load
    if os.path.exists("pickles/" + load_pickle):
        print("load pickle! for retrain")
        with open("pickles/" + load_pickle, "rb") as f:
            state_dict = torch.load(f, map_location=DEVICE)
        classifier.load_state_dict(state_dict)
        print("load pickle success")
    # TODO change of save_name
    save_name = "classifier_demo"
    viz = Visualizer(save_name, x_axis="step")
    dataset = CelebA("./celebA_image_png", usage="train", shuffle="yes")
    train(classifier, dataset, viz=viz, save_name=save_name, num_epoch=100, lr=1e-4)

    DEVICE2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier2 = Classifier().to(DEVICE2)
    print("start inference")
    # load_pickle = "classifier_100_leakrelu_sgd_epoch_99_0.0015756.pickle"
    load_pickle = glob.glob("pickles/"+ save_name+"_epoch_*.pickle")[-1]
    if os.path.exists( load_pickle):
        print("load pickle: " +load_pickle +"\n")
        with open(load_pickle, "rb") as f:
            state_dict = torch.load(f, map_location=DEVICE2)
        classifier2.load_state_dict(state_dict)

    dataset2 = CelebA("./celebA_image_png", usage="test", shuffle="yes")
    dataloader = torch.utils.data.DataLoader(dataset2, batch_size=64, shuffle=False)

    inference(classifier2, dataloader, save_name)