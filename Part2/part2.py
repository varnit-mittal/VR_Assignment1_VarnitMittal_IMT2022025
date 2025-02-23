import numpy as np
import imutils
import cv2
import os
import re

def detect_and_describe(image):
    """
    Detects keypoints and extracts feature descriptors from an image using SIFT.

    Parameters:
        image (numpy.ndarray): Input grayscale or color image.

    Returns:
        tuple:
            - kps (numpy.ndarray): Array of keypoints as (x, y) coordinates.
            - features (numpy.ndarray): Corresponding feature descriptors.
    """
    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute feature descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # Convert keypoints to NumPy array of (x, y) coordinates
    keypoints = np.float32([kp.pt for kp in keypoints])

    return keypoints, descriptors

def match_interest_points(keypointsA, keypointsB, descriptorsA, descriptorsB, ratio=0.75, reproj_thresh=5.0):
    """
    Matches keypoints between two images using the BFMatcher with KNN and filters good matches using Lowe's ratio test.
    Computes the homography matrix if enough matches are found.

    Parameters:
        keypointsA (numpy.ndarray): Keypoints (x, y) from the first image.
        keypointsB (numpy.ndarray): Keypoints (x, y) from the second image.
        descriptorsA (numpy.ndarray): Feature descriptors from the first image.
        descriptorsB (numpy.ndarray): Feature descriptors from the second image.
        ratio (float, optional): Lowe's ratio for filtering matches. Defaults to 0.75.
        reproj_thresh (float, optional): RANSAC reprojection threshold. Defaults to 5.0.

    Returns:
        tuple or None:
            - matches (list): List of filtered matches as index pairs.
            - homography (numpy.ndarray): 3x3 Homography matrix mapping image A to image B.
            - status (numpy.ndarray): Mask of inliers and outliers from RANSAC.
        Returns None if not enough matches are found.
    """
    # Initialize Brute-Force Matcher
    matcher = cv2.BFMatcher()

    # Find matches using KNN (K=2)
    raw_matches = matcher.knnMatch(descriptorsA, descriptorsB, k=2)
    matches = []

    # Apply Lowe's ratio test
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            matches.append((m.trainIdx, m.queryIdx))

    # Compute homography if sufficient matches are found
    if len(matches) > 4:
        ptsA = np.float32([keypointsA[i] for (_, i) in matches])
        ptsB = np.float32([keypointsB[i] for (i, _) in matches])
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thresh)
        return matches, H, status

    return None 

def draw_matches(imgA, imgB, kpA, kpB, matches, status):
    """Draws matched keypoints between two images."""
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]
    viz = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    viz[:hA, :wA] = imgA
    viz[:hB, wA:] = imgB

    for ((trainIdx, queryIdx), s) in zip(matches, status):
        if s == 1:
            ptA = (int(kpA[queryIdx][0]), int(kpA[queryIdx][1]))
            ptB = (int(kpB[trainIdx][0]) + wA, int(kpB[trainIdx][1]))
            cv2.line(viz, ptA, ptB, (0, 255, 0), 1)

    return viz

def crop_black_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return image[y:y + h - 1, x:x + w - 1]
    return image


def stitch(images, ratio=0.75, re_proj=5.0, show_overlay=False):
    """
    Stitches two images into a panorama using feature matching and homography.
    
    Args:
        images (tuple): A pair of images to be stitched.
        ratio (float): Ratio for feature matching (default is 0.75).
        re_proj (float): Threshold for re-projection error (default is 5.0).
        show_overlay (bool): If True, displays feature matches overlay (default is False).
        
    Returns:
        numpy.ndarray: The stitched panorama image, or a tuple of panorama and overlay if show_overlay is True.
    """
    imageA, imageB = images
    interestA, xA = detect_and_describe(imageA)
    interestB, xB = detect_and_describe(imageB)
    
    M = match_interest_points(interestA, interestB, xA, xB, ratio, re_proj)
    if M is None:
        print("Not enough matches found.")
        return None
    
    matches, H, status = M
    pano_img = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

    # Adjust height mismatch between images
    if pano_img.shape[0] != imageB.shape[0]:
        new_height = min(pano_img.shape[0], imageB.shape[0])
        imageB = cv2.resize(imageB, (imageB.shape[1], new_height))
        pano_img = pano_img[:new_height, :]
    
    pano_img[:imageB.shape[0], :imageB.shape[1]] = imageB
    pano_img = crop_black_region(pano_img)

    if show_overlay:
        overlay = draw_matches(imageA, imageB, interestA, interestB, matches, status)
        return pano_img, overlay

    return pano_img

def show(img, time=3, msg="Image"):
    cv2.imshow(msg, img)
    cv2.waitKey(int(time * 1000))
    cv2.destroyAllWindows()

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def main(input_dir, output_dir):
    """
    Creates a panorama by stitching images from the specified input directory and 
    saves the intermediate and final results in the output directory.

    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where stitched images will be saved.

    Raises:
        AssertionError: If no images are found in the input directory.
    """
    img_path = sorted([os.path.join(input_dir, i) for i in os.listdir(input_dir)], key=extract_number)
    assert img_path, "No image found in input folder"

    os.makedirs(output_dir, exist_ok=True)

    left_img = imutils.resize(cv2.imread(img_path[0]), width=600)

    for i, path in enumerate(img_path[1:], start=1):
        right_img = imutils.resize(cv2.imread(path), width=600)
        pano_img = stitch([left_img, right_img], show_overlay=True)
        if pano_img:
            left_img, viz = pano_img
            cv2.imwrite(os.path.join(output_dir, f"stitched_{i}.jpg"), viz)

    cv2.imwrite(os.path.join(output_dir, "panorama.jpg"), left_img)

if __name__ == "__main__":
    input_dir =  input("Enter the input directory: ")
    output_dir = input("Enter the output directory: ")
    
    main(input_dir, output_dir)
