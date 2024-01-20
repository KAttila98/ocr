from pytesseract import Output
import pytesseract
import numpy as np 
import cv2
import os
import matplotlib.pyplot as plt
import copy
from nms import nms
import math
from typing import Dict, List

class OCRModel():

    def __init__(self, confidence_threshold : int = 80, img_rotation_angle_range : range = range(-45,45,5)):

        self.confidence_threshold = confidence_threshold
        self.img_rotation_angle_range = img_rotation_angle_range
    
    @staticmethod
    def load_and_process_image(img_path : str, apply_denoising : bool = True, apply_thresholding : bool = True, img_resize_factor : int = 2, save_result_path : str = None) -> np.ndarray:

        # Based on: https://tesseract-ocr.github.io/tessdoc/ImproveQuality
        try:
            img = cv2.imread(img_path)
        except: 
            raise("Could not read image")
            
        img = cv2.resize(img, (img.shape[1]*img_resize_factor, img.shape[0]*img_resize_factor))

        if apply_denoising:
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)

        dominant_color = np.mean(img)

        # Apply this only for images with dark background (thus low text - background contrast)
        if dominant_color < 100:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
            # Here we are defining range of bluecolor in HSV 
            lower_blue = np.array([110,50,50]) 
            upper_blue = np.array([130,255,255])
            # This creates a mask of blue coloured  
            # objects found in the frame. 
            mask = cv2.inRange(hsv, lower_blue, upper_blue) 
            # The bitwise and of the frame and mask is done so  
            # that only the blue coloured objects are highlighted  
            # and stored in res 
            img = cv2.bitwise_and(img,img, mask= mask) 
            apply_thresholding = True
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # threshold the image using Otsu's thresholding method
        if apply_thresholding:
            img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        if save_result_path:
            plt.figure(figsize=(20,12))
            plt.imsave(save_result_path, img, cmap='gray')
        
        return img
    
    @staticmethod
    def __rotate(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy 
    
    @staticmethod
    def __visualize_results(img_path : str, filtered_rrects : List, filtered_texts : List, save_path : str = None):

        try:
            img = cv2.imread(img_path)
        except: 
            raise("Could not read image")

        for rrect, text in zip(filtered_rrects, filtered_texts): 
            box = cv2.boxPoints(rrect)
            box = np.int0(box)
            cv2.drawContours(img,[box],0,(0,0,255),2)  
            cv2.putText(img, text, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

        plt.figure(figsize=(20,12))
        if save_path is None: 
            plt.imshow(img)
        else:
            plt.imsave(save_path, img, cmap='gray')

    @staticmethod
    def __save_results_in_txt(txt_path, filtered_rrects : List, filtered_texts : List):

        with open(txt_path, "w") as f:
            for i in range(len(filtered_rrects)):
                f.write(f"Found text: {filtered_texts[i]}; Center position (in pixels): ({filtered_rrects[i][0][0]},{filtered_rrects[i][0][1]}); Dimension (width, height): ({filtered_rrects[i][1][0]},{filtered_rrects[i][1][1]}); Orientation (degrees): {filtered_rrects[i][2]}\n")
    
    def __apply_tesseract(self, image : np.ndarray, angle : int, all_results : Dict[int, Dict[str, List]], img_resize_factor : int):

        # Applying Tesseract to extract text
        results = pytesseract.image_to_data(image, output_type=Output.DICT)

        # Saving the predictions that are over the specified threshold and contain Latin characters
        # The predictions are stored in a dictionary, where the key is the rotation angle. This is necessary in order to eliminate later double detections in different angles
        all_results[angle] = {"left": [], "top": [], "height": [], "width": [], "conf": [], "text": []}

        for i in range(0, len(results["text"])):
            x = results["left"][i]
            y = results["top"][i]

            w = results["width"][i]
            h = results["height"][i]

            text = results["text"][i]
            conf = int(results["conf"][i])

            if conf > self.confidence_threshold:
                text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                if text != "" and len(text) > 1:
                    all_results[angle]["left"].append(x//img_resize_factor)
                    all_results[angle]["top"].append(y//img_resize_factor)

                    all_results[angle]["width"].append(w//img_resize_factor)
                    all_results[angle]["height"].append(h//img_resize_factor)

                    all_results[angle]["conf"].append(conf)
                    all_results[angle]["text"].append(text)

    def __filter_double_detections(self, unfiltered_results : Dict[int, Dict[str, List]], image_centerX : int, image_centerY : int, img_resize_factor : int):

        rrects = []
        scores = []
        texts = []

        # Calculating the original center point (before eventual upscaling) of the image for rotation
        orig_img_centerX, orig_image_centerY = image_centerX // img_resize_factor, image_centerY // img_resize_factor
        # Transforming back the rotated bounding boxes to the original image
        for angle in self.img_rotation_angle_range:               
            result = unfiltered_results[angle]

            for i in range(0, len(result["text"])):
                x = result["left"][i]
                y = result["top"][i]

                w = result["width"][i]
                h = result["height"][i]

                text = result["text"][i]
                conf = result["conf"][i]

                rect_center_x = (2*x+w)//2
                rect_center_y = (2*y+h)//2
                qx, qy = OCRModel.__rotate((orig_img_centerX, orig_image_centerY),(rect_center_x, rect_center_y), math.radians(angle))
                rrect = ((int(qx),int(qy)), (w,h), angle)
                rrects.append(rrect)
                scores.append(conf)
                texts.append(text)

        # Applying Non-Maximum Suppression to remove double detection potentially caused by processing the image in various angles
        nms_res = nms.rboxes(rrects, scores, nms_algorithm=nms.felzenszwalb.nms)   
        filtered_rrects = [rrects[i] for i in nms_res]
        filtered_texts = [texts[i] for i in nms_res]

        return filtered_rrects, filtered_texts
    
    def extract_text_from_image(self, img_path : str, apply_denoising : bool = True, apply_thresholding : bool = True, img_resize_factor : int = 2, txt_output_path : str = None, visualize_results : bool = False, output_image_path : str = None):

        img = OCRModel.load_and_process_image(img_path, apply_denoising, apply_thresholding, img_resize_factor)
        
        # Needed for image rotation
        height, width = img.shape[:2]
        centerX, centerY = (width // 2, height // 2)

        all_results = {}
        # Applying Tesseract algorithm on rotated image can help to perform better on images containing texts with various orientation
        for angle in self.img_rotation_angle_range:
            image = copy.deepcopy(img)

            # Rotating the image with specific angle
            M = cv2.getRotationMatrix2D((centerX, centerY), angle, 1.0)
            image = cv2.warpAffine(image, M, (width, height))

            # Applying Tesseract and adding the results to all_results dictionary
            all_result = self.__apply_tesseract(image, angle, all_results, img_resize_factor)
            
        
        filtered_rrects, filtered_texts = self.__filter_double_detections(all_results, centerX, centerY, img_resize_factor) 

        if txt_output_path:
            OCRModel.__save_results_in_txt(txt_output_path, filtered_rrects, filtered_texts)

        if visualize_results:
            OCRModel.__visualize_results(img_path, filtered_rrects, filtered_texts, output_image_path)