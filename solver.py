import cv2
import numpy as np
import easyocr
import base64
import re
import shutil
import tempfile
from functools import partial
from typing import Any, Union
import requests
import cv2
import numpy as np
import os
from touchtouch import touch
from PIL import Image
from tolerant_isinstance import isinstance_tolerant

reader = easyocr.Reader(["en"])


def _dummy_import():
    import torch


def get_tmpfile(suffix=".txt"):
    tfp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    filename = tfp.name
    filename = os.path.normpath(filename)
    tfp.close()
    return filename, partial(os.remove, tfp.name)


def save_cv_image(filepath: str, image: np.ndarray) -> str:
    format_ = filepath.split(".")[-1]
    filenametmp, deletefilename = get_tmpfile(suffix=f".{format_}")
    touch(path=filepath)
    touch(path=filenametmp)
    os.remove(filepath)
    cv2.imwrite(filenametmp, image)
    shutil.move(filenametmp, filepath)
    return filepath


def open_image_in_cv(
    image: Any, channels_in_output: Union[int, None] = None, bgr_to_rgb: bool = False
) -> np.ndarray:
    if isinstance(image, str):
        if os.path.exists(image):
            if os.path.isfile(image) or os.path.islink(image):
                try:
                    image2 = cv2.imread(image, cv2.IMREAD_UNCHANGED)
                    if isinstance_tolerant(image2, None):
                        image = np.array(Image.open(image))
                        bgr_to_rgb = not bgr_to_rgb
                    else:
                        image = image2
                except Exception:
                    try:
                        format_ = image.split(".")[-1]
                        filenametmp, deletefilename = get_tmpfile(suffix=f".{format_}")
                        try:
                            deletefilename()
                        except Exception:
                            pass
                        shutil.copy(image, filenametmp)
                        image = cv2.imread(filenametmp, cv2.IMREAD_UNCHANGED)
                        try:
                            deletefilename()
                        except Exception:
                            pass
                    except Exception:
                        image = np.array(Image.open(image))
                        bgr_to_rgb = not bgr_to_rgb

        elif re.search(r"^.{1,10}://", str(image)[:12]) is not None:
            x = requests.get(image).content
            if x.startswith(bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])):
                filenametmp, deletefilename = get_tmpfile(suffix=f".png")
                with open(filenametmp, mode="wb") as f:
                    f.write(x)
                image = cv2.imread(filenametmp, cv2.IMREAD_UNCHANGED)
                try:
                    deletefilename()
                except Exception:
                    pass
            else:
                image = cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_COLOR)
        else:
            try:
                image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                image = cv2.imdecode(
                    np.frombuffer(
                        base64.b64decode(image.split(",", maxsplit=1)[-1]), np.uint8
                    ),
                    cv2.IMREAD_COLOR,
                )
    elif isinstance(image, (bytes, bytearray)):
        image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    elif "PIL" in str(type(image)):
        image = np.array(image)
        bgr_to_rgb = not bgr_to_rgb

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if bgr_to_rgb:
        if len(image.shape) > 2:
            if image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            elif image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if channels_in_output is not None:
        if len(image.shape) > 2:
            if image.shape[-1] == 4 and channels_in_output == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            elif image.shape[-1] == 3 and channels_in_output == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            if image.shape[-1] == 4 and channels_in_output == 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif image.shape[-1] == 3 and channels_in_output == 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                pass
        else:
            if channels_in_output == 3 and len(image.shape) < 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if channels_in_output == 4 and len(image.shape) < 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)

    return image


def adjust_and_ocr(
    img: Any,
    thresh: int = 150,
    maxval: int = 255,
    cv2_type: int = cv2.THRESH_BINARY_INV,
    noise_width: int = 5,
    noise_height: int = 5,
) -> str:
    """
    Perform the following steps:
    1. Read the image (any format that can be read by PIL or OpenCV)
    2. Convert the image to grayscale
    3. Apply a binary threshold to the grayscale image using OpenCV's `cv2.threshold` with the given arguments
    4. Find contours using OpenCV's `cv2.findContours` and sort them by x-coordinate
    5. Create a new image with the same shape as the original, and iterate over the contours
    6. For each contour, if its width and height are both greater than the given noise width and height, copy the corresponding region from the original image to the new image, and increment the x-coordinate by the contour's width
    7. OCR the new image using EasyOCR and return the text

    :param img: The image to process
    :param thresh: The value to use for the binary threshold
    :param maxval: The maximum value to use for the binary threshold
    :param cv2_type: The type of thresholding to use (see OpenCV's `cv2.threshold` documentation)
    :param noise_width: The minimum width of a contour to consider it not noise
    :param noise_height: The minimum height of a contour to consider it not noise
    :return: The text that was extracted from the image
    """
    thresh = int(thresh)
    maxval = int(maxval)
    cv2_type = int(cv2_type)
    noise_width = int(noise_width)
    noise_height = int(noise_height)
    
    image = open_image_in_cv(img, channels_in_output=3)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(gray, thresh, maxval, cv2_type)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    image_copy = np.zeros_like(image)
    image_copy[:] = 255
    start_x = 1
    for ctr in sorted_contours:
        x, y, w, h = cv2.boundingRect(ctr)
        if w > noise_width and h > noise_height:
            image_copy[:h, start_x : start_x + w] = image[y : y + h, x : x + w]
            start_x += w
            start_x += 1

    result = reader.readtext(image_copy, detail=0)
    return result[0] if result else ""


def adjust_and_ocr_simple(
    img: Any,
    thresh: int = 150,
    maxval: int = 255,
    cv2_type: int = cv2.THRESH_BINARY_INV,
    noise_width: int = 5,
    noise_height: int = 5,
) -> str:
    """
    A simpler version that tries multiple approaches:
    1. Direct OCR on original image
    2. If that fails, try with different thresholding
    3. Fall back to contour-based method if needed
    """
    # Ensure parameters are the correct types
    thresh = int(thresh)
    maxval = int(maxval)
    cv2_type = int(cv2_type)
    noise_width = int(noise_width)
    noise_height = int(noise_height)
    
    image = open_image_in_cv(img, channels_in_output=3)
    
    print("Trying direct OCR on original image...")
    result = reader.readtext(image, detail=0)
    if result and any(result):
        print(f"Direct OCR succeeded: {result}")
        combined_result = ''.join(result)
        print(f"Combined result: {combined_result}")
        return combined_result
    
    print("Direct OCR failed, trying thresholding methods...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    threshold_methods = [
        (cv2.THRESH_BINARY_INV, "Binary INV"),
        (cv2.THRESH_BINARY, "Binary"),
        (cv2.THRESH_BINARY + cv2.THRESH_OTSU, "Otsu"),
        (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU, "Otsu INV")
    ]
    
    for method, name in threshold_methods:
        try:
            print(f"Trying {name} thresholding...")
            _, thresh_img = cv2.threshold(gray, thresh, maxval, method)
            result = reader.readtext(thresh_img, detail=0)
            if result and any(result):
                print(f"OCR worked with {name} thresholding: {result}")
                combined_result = ''.join(result)
                print(f"Combined result: {combined_result}")
                return combined_result
        except Exception as e:
            print(f"Error with {name} thresholding: {e}")
            continue
    
    print("Trying different threshold values...")
    for thresh_val in [100, 127, 150, 180, 200]:
        try:
            _, thresh_img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            result = reader.readtext(thresh_img, detail=0)
            if result and any(result):
                print(f"OCR worked with threshold {thresh_val}: {result}")
                combined_result = ''.join(result)
                print(f"Combined result: {combined_result}")
                return combined_result
        except Exception as e:
            print(f"Error with threshold {thresh_val}: {e}")
            continue
    
    print("Trying adaptive thresholding...")
    try:
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        result = reader.readtext(adaptive_thresh, detail=0)
        if result and any(result):
            print(f"OCR worked with adaptive thresholding: {result}")
            combined_result = ''.join(result)
            print(f"Combined result: {combined_result}")
            return combined_result
    except Exception as e:
        print(f"Error with adaptive thresholding: {e}")
    
    print("Trying morphological operations...")
    try:
        kernel = np.ones((2,2), np.uint8)
        
        morph_ops = [
            cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel),
            cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel),
            cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        ]
        
        for i, morph_img in enumerate(morph_ops):
            try:
                result = reader.readtext(morph_img, detail=0)
                if result and any(result):
                    print(f"OCR worked with morphological operation {i}: {result}")
                    combined_result = ''.join(result)
                    print(f"Combined result: {combined_result}")
                    return combined_result
            except:
                continue
    except Exception as e:
        print(f"Error with morphological operations: {e}")
    
        print("Trying different image preprocessing...")
    try:
        height, width = gray.shape
        resized = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        result = reader.readtext(resized, detail=0)
        if result and any(result):
            print(f"OCR worked with resized image: {result}")
            combined_result = ''.join(result)
            print(f"Combined result: {combined_result}")
            return combined_result
    except Exception as e:
        print(f"Error with image resizing: {e}")
    
    print("All preprocessing methods failed, trying contour method...")
    try:
        return adjust_and_ocr(img, thresh, maxval, cv2_type, noise_width, noise_height)
    except Exception as e:
        print(f"Contour method also failed: {e}")
        return ""


if __name__ == "__main__":
    try:
        print("Testing improved captcha solver...")
        
        result_simple = adjust_and_ocr_simple("captcha.jpg")
        print(f"Simple method result: '{result_simple}'")
        
        result_original = adjust_and_ocr("captcha.jpg")
        print(f"Original method result: '{result_original}'")
        
        print("\nSuccess! The captcha contains:", result_simple)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
