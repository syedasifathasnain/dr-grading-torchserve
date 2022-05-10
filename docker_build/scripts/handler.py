import logging
import torch
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms

# from serve.ts.torch_handler.base_handler import BaseHandler
from ts.torch_handler.image_classifier import ImageClassifier
import argparse
import numpy as np

from model import DRModel
from PIL import Image

import numpy as np
import torch
import cv2
import random

##### IMAGE PREPROCESSING

def prepare_image(img, 
                  sigmaX         = 10, 
                  do_random_crop = False):
    
    '''
    Preprocess image
    '''
    img = np.array(img) 
    # print(image)
    image = img[:, :, ::-1].copy() 
    # # import image
    # # image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # perform smart crops
    image = crop_black(image, tol = 7)
    if do_random_crop == True:
        image = random_crop(image, size = (0.9, 1))
    
    # resize and color
    image = cv2.resize(image, (int(299), int(299)))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    
    # circular crop
    image = circle_crop(image, sigmaX = sigmaX)

    # convert to tensor    
    # image = torch.tensor(image)
    # image = image.permute(2, 1, 0)
    return image



##### CROP FUNCTIONS

def crop_black(img, 
               tol = 7):
    
    '''
    Perform automatic crop of black areas
    '''
    print(img.ndim)
    # print(np.shape(img))
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        
        if (check_shape == 0): 
            return img 
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img  = np.stack([img1, img2, img3], axis = -1)
            return img
        
        
def circle_crop(img, 
                sigmaX = 10):   
    
    '''
    Perform circular crop around image center
    '''
        
    height, width, depth = img.shape
    
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape
    
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness = -1)
    
    img = cv2.bitwise_and(img, img, mask = circle_img)
    return img 


def crop_for_squares(img, 
                     cut = 0.1):
    
    '''
    Crop for square images
    '''

    if (max(img.shape[0:2]) / min(img.shape[0:2]) < 1.1):
        h = max(img.shape)
        if (img.shape[0] == h):
            img = img[int(cut*h):int((1-cut)*h), :, :]
        else:
            img = img[:, int(cut*h):int((1-cut)*h), :]
            
    return img


def standard_crop(img, 
                  cut = 0.1):
    
    '''
    Standard crop
    '''

    height, width, depth = img.shape    
        
    img = img[int(cut*height):int((1-cut)*height), int(cut*width):int((1-cut)*width), :]
            
    return img



def random_crop(img, 
                size = (0.9, 1)):
    
    '''
    Random crop
    '''

    height, width, depth = img.shape
    
    cut = 1 - random.uniform(size[0], size[1])
    
    i = random.randint(0, int(cut * height))
    j = random.randint(0, int(cut * width))
    h = i + int((1 - cut) * height)
    w = j + int((1 - cut) * width)

    img = img[i:h, j:w, :]    
    
    return img



##### CUTOUT AUGMENTATION

class cutout(object):
    
    '''
    Cutout augmentation
    '''

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length  = length

    def __call__(self, img):

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
    
    
##### STATS CALCULATION
    
def online_mean_and_sd(loader):
    
    '''
    Calculate mean and SD
    '''

    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:

        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim = [0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim = [0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


class MyHandler(ImageClassifier):
    """
    Custom handler for pytorch serve. This handler supports batch requests.
    For a deep description of all method check out the doc:
    https://pytorch.org/serve/custom_service.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = transforms.Compose([
                    # transforms.Resize((299, 299)),
                    transforms.ToPILImage(),
                    transforms.ToTensor()
                    # transforms.Normalize((0.485, 0.456, 0.406),
                    #                     (0.229, 0.224, 0.225))])
        ])
        self.model = DRModel.get_model(5)
        #print(self.model)

        device = torch.device('cpu')

        # self.model.load_state_dict(torch.load('best.pth',map_location=torch.device('cpu')))
        self.model.load_state_dict(torch.load('model_enet_b4.bin',map_location=torch.device('cpu')))

        self.model.eval()
        self.model = self.model.to(device)
        
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.initialized = True
        #  load the model, refer 'custom handler class' above for details
        

    # def preprocess_one_image(self, req):
    #     """
    #     Process one single image.
    #     """
    #     # get image from the request
    #     image = req.get("data")
    #     if image is None:
    #         image = req.get("body")       
    #      # create a stream from the encoded image
    #     image = Image.open(io.BytesIO(image))
    #     image = self.transform(image)
    #     # add batch dim
    #     image = image.unsqueeze(0)
    #     return image

    # def predict_image(img, model):
    #     labels = {0 : 'No DR',1 : 'Mild', 2 : 'Moderate',3 : 'Severe',4 : 'Proliferative DR'}
        
    #     # Convert to a batch of 1
    #     xb = img.unsqueeze(0).cuda()
    #     # Get predictions from model
    #     yb = model(xb)
    #     # Pick index with highest probability
    #     prob, preds  = torch.max(yb, dim=1)
    #     # Retrieve the class label
    #     # return labels[preds[0].item()]
    #     return prob, preds

    def preprocess(self, requests):
        """
        Process all the images from the requests and batch them in a Tensor.
        """
        # images = [self.preprocess_one_image(req) for req in requests]
        # images = torch.cat(images)    
        # return images
        
        image = requests[0].get("file")  #data-> [1.png,2.png]
        
        if image is None:
            image = requests[0].get("body")  
        # image = Image.open(image).convert('RGB')
        # image2 = Image.open(image2]).convert('RGB')
  
        image = Image.open(io.BytesIO(image)).convert('RGB')
        print(image)
        image = prepare_image(image)#cv2.imread(img_fname)

        image = self.transform(image)

        # image = torch.stack((image1, image2))
        image = torch.unsqueeze(image, 0)
        print(image.shape)
        device = torch.device('cpu')
        image = image.to(device)

        return image


    def inference(self, image):
        """
        Given the data from .preprocess, perform inference using the model.
        We return the predicted label for each image.
        """
        # outs = self.model.forward(x)
        # probs = F.softmax(outs, dim=1) 
        # preds = torch.argmax(probs, dim=1)
        # return preds
        
        prediction = self.model(image)

        return prediction

    def postprocess(self, prediction):
        """
        Given the data from .inference, postprocess the output.
        In our case, we get the human readable label from the mapping 
        file and return a json. Keep in mind that the reply must always
        be an array since we are returning a batch of responses.
        """
        # # res = []
        # # # pres has size [BATCH_SIZE, 1]
        # # # convert it to list
        # # preds = preds.cpu().tolist()
        # # for pred in preds:
        # #     label = self.mapping[str(pred)][1]
        # #     res.append({'label' : label, 'index': pred })
        # # return res

        # return self.model.tokenizer.decode_batch(prediction.cpu().numpy())
        labels = {0 : 'No DR',1 : 'Mild', 2 : 'Moderate',3 : 'Severe',4 : 'Proliferative DR'}
        
        # Convert to a batch of 1
        # xb = img.unsqueeze(0).cuda()
        # # Get predictions from model
        # yb = model(xb)
        # Pick index with highest probability
        
        prob, preds  = torch.max(prediction, dim=1)
        print(prediction, preds[0].item())
        # prediction  = prediction.detach().numpy().tolist()[0]
        # print([prediction])
        # print(prediction.detach().numpy().tolist())
        # dr_grade = {}
        # for i in range(0,5):
        #     dr_grade[labels[i]] = prediction[i]
        # print("prob pred: ", prob, preds)
        # Retrieve the class label
        return [{"dr-score": preds[0].item()}]
        # return [prediction]