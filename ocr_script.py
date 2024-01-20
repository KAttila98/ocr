import pytesseract
import os
import argparse
from model  import OCRModel
from tqdm import tqdm


if __name__ == '__main__':

    # EXAMPLE COMMAND: python ocr_script.py --image_folder_path testfiles --tesseract_exe_path 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' --apply_denoising --apply_thresholding --image_upscale_factor 2 --save_visualizations
    
    parser = argparse.ArgumentParser(description='This script performs OCR on image input to extract text from it')
    parser.add_argument('--image_folder_path', type=str,
                    help='Path to the folder containing the images', required=True)
    parser.add_argument('--tesseract_exe_path', type=str,
                    help='Path to the tesseract engine', required=True)
    parser.add_argument('--visualization_folder_path', type=str,
                    help='Path to the folder where visualizations (if desired) will be saved', default="visualizations")
    parser.add_argument('--text_output_folder_path', type=str,
                    help='Path to the folder where extracted text will be saved in form of .txt files', default="extracted_text")
    parser.add_argument('--image_upscale_factor', type=int,
                    help='If input images have low resolution, this should be set to an integer value higher than 1 in order to upscale the images', default=2)  
    parser.add_argument('--apply_denoising', action="store_true",
                    help='When True, OpenCV denoising algorithm will be applied on the images') 
    parser.add_argument('--apply_thresholding', action="store_true",
                    help='When True, OpenCV OTSU thresholding algorithm will be applied on the images')
    parser.add_argument('--save_visualizations', action="store_true",
                    help='When True, visualizations containing the highlighted, recognized text will be saved into visualization_folder_path')  
    args = parser.parse_args()

    pytesseract.pytesseract.tesseract_cmd = args.tesseract_exe_path

    os.makedirs(args.text_output_folder_path, exist_ok=True)
    os.makedirs(args.visualization_folder_path, exist_ok=True)

    filenames = os.listdir(args.image_folder_path)

    for filename in tqdm(filenames):
        ocr = OCRModel()
        ocr.extract_text_from_image(img_path = f'{args.image_folder_path}/{filename}', 
                                    apply_denoising = args.apply_denoising, apply_thresholding = args.apply_thresholding, img_resize_factor = args.image_upscale_factor, 
                                    visualize_results = args.save_visualizations, output_image_path = f'{args.visualization_folder_path}/{filename.split(".")[0]}.png', 
                                    txt_output_path = f'{args.text_output_folder_path}/{filename.split(".")[0]}_extracted.txt')