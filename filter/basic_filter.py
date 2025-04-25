# Import the Image class for basic image operations and ImageFilter for built-in filters
from PIL import Image, ImageFilter

# Import pyplot from Matplotlib for displaying images and saving figures
import matplotlib.pyplot as plt  

# Import OpenCV for image I/O and processing
import cv2

# Import os for filesystem path operations
import os

def apply_blur_filter(image_path):
    """Resize to 128×128, apply Gaussian blur (radius=2), display and save."""
    try:
        img = Image.open(image_path)                             # open image file
        img_resized = img.resize((128, 128))                     # resize to 128×128 pixels
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))  # apply Gaussian blur

        plt.imshow(img_blurred)                                  # display blurred image
        plt.axis('off')                                          # remove axes
        plt.savefig("blurred_image.png")                         # save figure to file
        print("Saved blurred_image.png")                         # confirm save
    except Exception as e:
        print(f"Error: {e}")                                     # print any error

def apply_sharpen_filter(image_path):
    """Resize to 128×128, apply sharpen filter, display and save."""
    try:
        img = Image.open(image_path)                             # open image file
        img_resized = img.resize((128, 128))                     # resize to 128×128 pixels
        img_sharp = img_resized.filter(ImageFilter.SHARPEN)      # apply sharpen filter

        plt.imshow(img_sharp)                                    # display sharpened image
        plt.axis('off')                                          # remove axes
        plt.savefig("sharpened_image.png")                      # save figure to file
        print("Saved sharpened_image.png")                       # confirm save
    except Exception as e:
        print(f"Error: {e}")                                     # print any error

def apply_edge_filter(image_path):
    """Resize to 128×128, apply edge detection, display and save."""
    try:
        img = Image.open(image_path)                             # open image file
        img_resized = img.resize((128, 128))                     # resize to 128×128 pixels
        img_edges = img_resized.filter(ImageFilter.FIND_EDGES)   # apply edge detection filter

        plt.imshow(img_edges)                                    # display edge-detected image
        plt.axis('off')                                          # remove axes
        plt.savefig("edged_image.png")                           # save figure to file
        print("Saved edged_image.png")                           # confirm save
    except Exception as e:
        print(f"Error: {e}")                                     # print any error

def apply_emboss_filter(image_path):
    """Resize to 128×128, apply emboss filter, display and save."""
    try:
        img = Image.open(image_path)                             # open image file
        img_resized = img.resize((128, 128))                     # resize to 128×128 pixels
        img_emboss = img_resized.filter(ImageFilter.EMBOSS)      # apply emboss filter

        plt.imshow(img_emboss)                                   # display embossed image
        plt.axis('off')                                          # remove axes
        plt.savefig("embossed_image.png")                        # save figure to file
        print("Saved embossed_image.png")                        # confirm save
    except Exception as e:
        print(f"Error: {e}")                                     # print any error

def swap_halves(img_path, direction):
    """
    Split the image in half and swap the two parts.
    
    - img_path: path to the input image
    - direction: 'vertical' (left/right) or 'horizontal' (top/bottom)
    - output_suffix: suffix for the output filename
    """
    # Read the image from disk in BGR format
    img = cv2.imread(img_path)
    # If loading failed, raise an error
    if img is None:
        raise FileNotFoundError(f"Cannot load image at {img_path}")
    
    # Get the image height (h) and width (w)
    h, w = img.shape[:2]
    
    if direction == 'vertical':
        # Extract left half: all rows, columns from 0 to w//2
        left  = img[:, : w//2]
        # Extract right half: all rows, columns from w//2 to end
        right = img[:, w//2 :]
        # Concatenate right half first, then left half horizontally
        swapped = cv2.hconcat([right, left])
        # Static filename for the vertical split output
        out_name = "vertical-split_image.jpg"
        # Write the swapped image to disk
        cv2.imwrite(out_name, swapped)
        # Print a confirmation message
        print(f"Saved {out_name}")
    else:  # horizontal swap
        # Extract top half: rows from 0 to h//2, all columns
        top    = img[: h//2, :]
        # Extract bottom half: rows from h//2 to end, all columns
        bottom = img[h//2 :, :]
        # Concatenate bottom half first, then top half vertically
        swapped = cv2.vconcat([bottom, top])
        # Static filename for the horizontal split output
        out_name = "horizontal-split_image.jpg"
        # Write the swapped image to disk
        cv2.imwrite(out_name, swapped)
        # Print a confirmation message
        print(f"Saved {out_name}")

    return out_name

if __name__ == "__main__":
    image_path = "avatar-256.jpg"                                # path to your input image
    apply_blur_filter(image_path)                                # run blur filter
    apply_sharpen_filter(image_path)                             # run sharpen filter
    apply_edge_filter(image_path)                                # run edge detection filter
    apply_emboss_filter(image_path)                              # run emboss filter
    swap_halves(image_path, direction='vertical')          # run image split (vertical) filter
    swap_halves(image_path, direction='horizontal')        # run image split (hotizontal) filter
