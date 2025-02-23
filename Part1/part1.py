import cv2
import numpy as np
import os
import shutil

def showSave(img,counter):
    """
    This function displays the image and saves the image.

    Parameters:
        img (numpy.ndarray): Image to be displayed.
        counter (int): Counter for the image name.
    """
    cv2.imshow("Image", img)
    cv2.waitKey(int(3*1000))
    cv2.destroyAllWindows()


    save_folder = "output"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if(counter):
        cv2.imwrite(os.path.join(save_folder,f"Segmented_Coin_{counter}.jpg"),img)
    else:
        cv2.imwrite(os.path.join(save_folder,"Processed_Coin.jpg"),img)

def preprocess_image(path: str):
    """
    Loads, resizes, and preprocesses an image for analysis.
    
    Parameters:
        path (str): Path to the image file.
    
    Returns:
        tuple: Processed image, thresholded image, and scale factor.
    """
    image = cv2.imread(path)
    scale_factor = 700 / max(image.shape[:2])
    image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    showSave(thresh, 0)
    return image, thresh, scale_factor

def detect_edges(image, threshold, scale):
    """
    Detects and draws circular contours in a thresholded image.
    
    Parameters:
        image (numpy.ndarray): Original image.
        threshold (numpy.ndarray): Thresholded image.
        scale (float): Scale factor for minimum area.
    
    Returns:
        list: Circular contours.
    """
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_contours = [cnt for cnt in contours if (perimeter := cv2.arcLength(cnt, True)) > 0 
                         and 0.7 < (circularity := 4 * np.pi * (cv2.contourArea(cnt) / (perimeter ** 2))) < 1.2
                         and cv2.contourArea(cnt) > 500 * (scale ** 2)]
    
    cv2.drawContours(image, detected_contours, -1, (0, 255, 0), 2)
    return detected_contours

def segment_coins(img, contours):
    """
    Segments circular objects from an image using detected contours.
    
    Parameters:
        img (numpy.ndarray): Original image.
        contours (list): List of detected contours.
    
    Returns:
        list: List of segmented coin images.
    """
    segmented_coins = []
    
    for i, cnt in enumerate(contours):
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center, radius = (int(x), int(y)), int(radius)
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        
        coin_segment = cv2.bitwise_and(img, mask)[center[1] - radius:center[1] + radius, center[0] - radius:center[0] + radius]
        segmented_coins.append(coin_segment)
        showSave(coin_segment, i + 1)
    
    return segmented_coins


def count_coin(contours,coins):
    return (len(contours),len(coins))

def main(path="coins1.jpg"):
    image,thresh,scale=preprocess_image(path)
    contours=detect_edges(image,thresh,scale)
    coins=segment_coins(image,contours)
    return count_coin(contours,coins)

if __name__ == "__main__":
    input_folder = "input"
    save_folder = "output"

    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)

    inp=input("Enter the image number (1-4): ")
    main(os.path.join(input_folder,f"coins{inp}.jpg"))