import numpy as np
from abc import ABC, abstractmethod

from optparse import OptionParser
import os

import cv2


def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape)/2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
  return result


def rotateImage90(image):
    w = image.shape[0]
    h = image.shape[1]
    M = cv2.getRotationMatrix2D((w/2, h/2), 90, 1.0)
    return cv2.warpAffine(image, M, (h, w))


def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]

    return img_crop


class CardRecognizer(ABC):
    @abstractmethod
    def train(self, train_images_labels):
        ''' 
            Assumes perfectly aligned training images.
        '''
        return None

    @abstractmethod
    def recognize_cards(self, table_image):
        return None


class CR_TemplateMatching(CardRecognizer):
    def __init__(self, templates, threshold=0.75):
        self.threshold = threshold
        self.templates = templates

    def train(self, train_images_labels):
        '''
            Thresholds should be set here for each class.
        '''
        None
    
    def recognize_cards(self, table_image):
        '''
            1. Compute gaussian pyramid.
               For each image in the pyramid:
            2. Detect lines in the image and rotate it such that lines are axis-parallel.
            3. Perform template matching using threshold (problematic for occluded cards).

            NEXT STEP: Correct bounding boxes after adaptive threshold and contour approximation.
                       Either extend bounding boxes or use subimage of templates to match.
        '''
        gray_templates = {}
        for label in self.templates.keys():
            gray_templates[label] = cv2.cvtColor(self.templates[label][0], cv2.COLOR_BGR2GRAY)

        angle_resolution = 360
        recognized_cards = []
        img_gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
        img_gray2 = cv2.GaussianBlur(img_gray,(5,5),5)
        flag, thresh = cv2.threshold(img_gray2, 120, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        connectivity = 8  # Could also be 4.
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        template_height, template_width = self.templates['G0'][0].shape[:2]
        print(template_height, template_width)
        # Contour area should at least be as large as the size of the training images to get a match.
        for contour in contours:
            #x,y,w,h = cv2.boundingRect(contour)  # minAreaRect
            #if w * h >= template_height * template_width / 4:
            #    cv2.rectangle(table_image,(x,y),(x+w,y+h),(0,255,0),2)
            #    cv2.minAreaRect(contour)
                #cv2.imwrite(str(x) + '_' + str(y) + '.jpg', table_image[y:y+h, x:x+w])
                #cv2.imwrite(str(x) + '_' + str(y) + '.jpg', crop_minAreaRect(table_image, cv2.minAreaRect(contour)))
            minAreaRect = cv2.minAreaRect(contour)
            #print(minAreaRect)
            w = minAreaRect[1][0]
            h = minAreaRect[1][1]
            if w > 10 and h > 10:
                cropped = crop_minAreaRect(img_gray, minAreaRect)
                #w, h = cropped.shape[::-1]
                print(w, h, cropped.shape)
                if w > h:
                    cropped = rotateImage90(cropped)
                cv2.imshow('cropped', cropped)
                cv2.waitKey(0)
                #print(w, h)
                print(cropped.shape)
                cropped = cv2.resize(cropped, (template_width, template_height))
                for label in self.templates.keys():
                    template = gray_templates[label]
                    res = cv2.matchTemplate(cropped, template, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= self.threshold)
                    if len(loc[0]) != 0:
                        recognized_cards.append(label)
        return set(recognized_cards)
        cv2.imwrite('out.jpg', table_image)
        cv2.waitKey(0)
        return set()
        #diff = cv2.dilate(diff, np.ones((5,5), np.uint8))
        cv2.imwrite('out.jpg', diff)
        for angle in range(int(360/angle_resolution)):
            print(angle)
            img = rotateImage(img_gray, angle * angle_resolution)
            for label in self.templates.keys():
                template = gray_templates[label]
                w, h = template.shape[::-1]
                res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= self.threshold)
                if len(loc[0]) != 0:
                    recognized_cards.append(label)
        return set(recognized_cards)




if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--folder', action='store', type='string', dest='folder',
                      help='The folder with training images.', default='training_images')
    parser.add_option('--test-image', action='store', type='string', dest='test_image', help='')
    (options, args) = parser.parse_args()

    # Load training images and fill the training dict.
    training_tuples = []
    training_files = os.listdir(options.folder)
    for train_file in training_files:
        training_tuples.append((train_file, cv2.imread(os.path.join(options.folder, train_file), cv2.IMREAD_COLOR)))
    train_images_labels = {}
    train_label_set = set([f[0][:-4] for f in training_tuples])
    for train_file in training_tuples:
        label = train_file[0][:-4]
        files_for_label = []
        if label in train_images_labels.keys():
            files_for_label = train_images_labels[label]
        files_for_label.append(train_file[1])
        train_images_labels[label] = files_for_label
    
    recognizer = CR_TemplateMatching(train_images_labels)
    # There must be a function that recognizes all cards on the table.
    cards_on_table = set(os.path.splitext(os.path.basename(options.test_image))[0].split('_'))
    recognized_cards = recognizer.recognize_cards(cv2.imread(options.test_image, cv2.IMREAD_COLOR))
    print(' Correctly recognized')
    print((recognized_cards & cards_on_table))
    print(' False positives (recognized, but not present)')
    print((recognized_cards - cards_on_table))
    print(' True negatives (not recognized, but present)')
    print((cards_on_table - recognized_cards))
