import pytesseract
import os
import argparse
from model  import OCRModel
from tqdm import tqdm
import sys


if __name__ == '__main__':

    # EXAMPLE COMMAND: python ocr_script.py --image_folder_path testfiles --tesseract_exe_path "C:\\Program Files\\Tesseract-OCR\\tesseract.exe" --confidence_threshold 80 --rotation_range -45 45 5 --apply_denoising --thresholding_method adaptive --image_upscale_factor 2 --save_visualizations
    
    parser = argparse.ArgumentParser(description='This script performs OCR on image input to extract text from it')
    parser.add_argument('--image_folder_path', type=str,
                    help='Path to the folder containing the images', required=True)
    parser.add_argument('--tesseract_exe_path', type=str,
                    help='Path to the tesseract engine. It is required when using the script on Windows')
    parser.add_argument('--visualization_folder_path', type=str,
                    help='Path to the folder where visualizations (if desired) will be saved', default="visualizations_")
    parser.add_argument('--text_output_folder_path', type=str,
                    help='Path to the folder where extracted text will be saved in form of .txt files', default="extracted_text_")
    parser.add_argument('--image_upscale_factor', type=int,
                    help='If input images have low resolution, this should be set to an integer value higher than 1 in order to upscale the images', default=2)  
    parser.add_argument('--apply_denoising', action="store_true",
                    help='When True, OpenCV denoising algorithm will be applied on the images') 
    parser.add_argument('--thresholding_method', type=str,
                    help='Defines which thresholding algorithm will be applied on the images', choices=[None, "otsu", "adaptive"], default=None)
    parser.add_argument('--confidence_threshold', type=int,
                    help='Predictions above this threshold will be saved and forwarded to NMS', default=80)  
    parser.add_argument('--rotation_range',
                    help='Range of image rotation angles in format of: from_angle to_angle step_size', nargs='+', default=[-45,45,5])  
    parser.add_argument('--save_visualizations', action="store_true",
                    help='When added, visualizations containing the highlighted, recognized text will be saved into visualization_folder_path')  
    args = parser.parse_args()

    # Path to tesseract engine is required when using the script on WIndows
    if "win" in sys.platform:
        if args.tesseract_exe_path:
            pytesseract.pytesseract.tesseract_cmd = args.tesseract_exe_path
        else:
            raise Exception("Tesseract engine path is required when using the script on Windows")
            
    # Creating output folders
    os.makedirs(args.text_output_folder_path, exist_ok=True)
    os.makedirs(args.visualization_folder_path, exist_ok=True)

    # Initializing OCR Model
    filenames = os.listdir(args.image_folder_path)
    ocr = OCRModel(confidence_threshold=args.confidence_threshold, img_rotation_angle_range=range(int(args.rotation_range[0]), int(args.rotation_range[1]), int(args.rotation_range[2])))

    # For each image under image_folder_path perform text extraction
    for filename in tqdm(filenames):
        ocr.extract_text_from_image(img_path = f'{args.image_folder_path}/{filename}', 
                                    apply_denoising = args.apply_denoising, thresholding_method = args.thresholding_method, img_resize_factor = args.image_upscale_factor, 
                                    visualize_results = args.save_visualizations, output_image_path = f'{args.visualization_folder_path}/{filename.split(".")[0]}.png', 
                                    txt_output_path = f'{args.text_output_folder_path}/{filename.split(".")[0]}_extracted.txt')