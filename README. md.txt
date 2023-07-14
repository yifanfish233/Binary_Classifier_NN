To run the classification:

prequest: need Nvidia GPU with function well of cuda and realated library.
 
First startup the visdom server in a terminal “python -m visdom.server”

Second, run  “python3 main.py”, this will run the train and inference together.

In the inference part, we will get classifier_feature_record.txt, which is what we need for later.

Once we get pickle file, we can use the model to infer the attribute.

If need to run the original data, u need to download from here: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

By changing the code of the path in the main.py, we need also need change the size we are reading in the files in the dataloader.py by uncommon the necessary codes.

