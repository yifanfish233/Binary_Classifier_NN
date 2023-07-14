import os
import torch
from visdom import Visdom
import time
import numpy as np


# python -m visdom.server

# python -m visdom.server -p 9098

# cmd = sys.executable + " -m visdom.server"
# subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True)


# http://localhost:8097/

class Visualizer:
    def __init__(self, env_name, port=8097, x_axis="step"):
        self.x_axis = x_axis
        self.vis = Visdom(port=port)
        self.__env = os.path.basename(env_name)

        self.__plot_windows = {}
        self.__paint_windows = {}

        self.__x_counter = 0

        # cmd = sys.executable + " -m visdom.server"
        # self.log_out_process = subprocess.Popen("exec " + cmd, stdout=subprocess.PIPE, shell=True)

    def every_n_step(self, n):
        return self.__x_counter % n == 0

    def tic(self):
        self.__x_counter += 1

    def plots(self, names, ys, sub_name=""):
        names = [name + sub_name for name in names]
        for name, y in zip(names, ys):
            self.plot(name, y)

    def imShows(self, names, ims, sub_name=""):
        names = [name + sub_name for name in names]
        for name, im in zip(names, ims):
            self.imShow(name, im)

    def cat_images(self, name, images, save_history=False):
        # image format float 0->1
        assert set([im.dtype for im in images]) == {torch.float32}
        images = [torch.clamp(im, 0, 1) for im in images]
        cat_image = torch.cat(images, dim=-1)  # dim=chw[-1]=w
        self.imShow(name, cat_image, save_history=save_history)

    def cat_batch_images(self, name, images, save_history=False, nrow=1):
        # image format float 0->1
        assert set([im.dtype for im in images]) == {torch.float32}
        images = [torch.clamp(im, 0, 1) for im in images]
        cat_image = torch.cat(images, dim=-1)  # dim=nchw[-1]=w
        self.batch_im_show(name, cat_image, save_history=save_history, nrow=nrow)

    def plot(self, name, y):
        y = y.item() if isinstance(y, torch.Tensor) else y
        x = self.__x_counter
        if len(name.split('/')) == 2:
            windowName, variableName = name.split('/')
        else:
            windowName, variableName = name, name
        if windowName not in self.__plot_windows:
            self.__plot_windows[windowName] = self.vis.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.__env,
                                                            opts=dict(
                                                                legend=[
                                                                    variableName],
                                                                title=windowName,
                                                                xlabel=self.x_axis,
                                                                ylabel=variableName
                                                            ))
        else:
            self.vis.line(X=np.array([x]), Y=np.array([y]), env=self.__env, win=self.__plot_windows[windowName],
                          name=variableName, update='append')

    def imShow(self, name, im, save_history=False):
        titleName = name + " " + self.x_axis + ": " + str(self.__x_counter)
        windowName = titleName if save_history else name

        if windowName not in self.__paint_windows:
            self.__paint_windows[windowName] = self.vis.image(
                im, env=self.__env, opts=dict(title=titleName))
        else:
            self.vis.image(im, env=self.__env, opts=dict(
                title=titleName), win=self.__paint_windows[windowName])

    def batch_im_show(self, name, batch_image, save_history=False, nrow=1):
        titleName = name + " " + self.x_axis + ": " + str(self.__x_counter)
        windowName = titleName if save_history else name

        if windowName not in self.__paint_windows:
            self.__paint_windows[windowName] = self.vis.images(
                batch_image, env=self.__env, opts=dict(title=titleName), nrow=nrow)
        else:
            self.vis.images(batch_image, env=self.__env, opts=dict(title=titleName),
                            win=self.__paint_windows[windowName], nrow=nrow)


if __name__ == '__main__':

    myvis = Visualizer("temp test22343542")
    for i in range(1, 1000):
        myvis.plot('loss/train', -9 * i)
        myvis.plot('loss/val', 3 * i)

        img = np.ones((3, 350, 350)) * (i % 5) * 50
        img = img.astype(np.uint8)
        myvis.imShow('testImage', img, save_history=True)

        myvis.tic()

        time.sleep(1)
        print(i)
