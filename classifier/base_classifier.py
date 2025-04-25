import os  # for file and path operations
import tensorflow as tf  # import TensorFlow library
tf.get_logger().setLevel('ERROR')  # suppress TensorFlow logs below ERROR level
from tensorflow.keras.applications import MobileNetV2  # import the MobileNetV2 architecture
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions  # import preprocessing & decoding utilities
from tensorflow.keras.preprocessing import image  # import image loading utilities
import numpy as np  # import NumPy for numerical operations
import matplotlib.pyplot as plt  # import Matplotlib for plotting (used earlier, not in occlusion)
import cv2  # import OpenCV for image I/O and processing (pip install opencv-python-headless)

# Instantiate the MobileNetV2 model with pre-trained ImageNet weights
model = MobileNetV2(weights="imagenet")

def classify_image(image_path):
    """Classify an image and display the predictions."""
    try:
        img = image.load_img(image_path, target_size=(224, 224))  # load & resize image
        img_array = image.img_to_array(img)  # convert PIL image to NumPy array
        img_array = preprocess_input(img_array)  # apply MobileNetV2 preprocessing
        img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

        predictions = model.predict(img_array)  # run inference
        decoded_predictions = decode_predictions(predictions, top=3)[0]  # decode top-3 ImageNet labels

        print("Top-3 Predictions:")  # header
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):  # iterate over results
            print(f"{i + 1}: {label} ({score:.2f})")  # print rank, label, and confidence

        return img_array, decoded_predictions  # return preprocessed tensor & labels

    except Exception as e:
        print(f"Error processing image: {e}")  # on error, print it
        return None, None  # return None pair if failure

def make_gradcam_heatmap(img_tensor, model, last_conv_layer_name, pred_index=None):
    """
    Generate a Grad-CAM heatmap for the given image tensor and model.
    Returns a 2D heatmap normalized to [0, 1].
    """
    grad_model = tf.keras.models.Model(  # create a model that gives conv layer outputs and predictions
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:  # record operations for automatic differentiation
        conv_outputs, predictions = grad_model(img_tensor)  # forward pass
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])  # choose top class if none specified
        class_channel = predictions[:, pred_index]  # isolate score for the target class

    grads = tape.gradient(class_channel, conv_outputs)  # compute gradients of score w.r.t. conv outputs
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # global average pooling of gradients

    conv_outputs = conv_outputs[0]  # drop batch dimension
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # weight feature maps by importance
    heatmap = tf.squeeze(heatmap)  # remove extra dims

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)  # apply ReLU & normalize
    return heatmap.numpy()  # return as NumPy array

def overlay_heatmap(img_path, heatmap, output_path=None, alpha=0.6):
    """
    Superimpose the heatmap onto the original image (heatmap on top),
    then save it to disk.
    """
    import os  # for path operations within this function

    img = cv2.imread(img_path)  # read the original image (BGR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image at {img_path}")  # error if file missing

    # Resize & convert heatmap to color
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # upscale to match image size
    heatmap_uint8   = np.uint8(255 * heatmap)  # scale to 0-255
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # apply JET colormap

    # Blend original image and heatmap
    superimposed = cv2.addWeighted(
        img,             # background image
        1 - alpha,       # background weight
        heatmap_colored, # heatmap overlay
        alpha,           # heatmap weight
        0                # no additional scalar
    )

    if output_path is None:
        base, ext = os.path.splitext(img_path)  # split name & extension
        output_path = f"{base}-heatmap{ext}"    # default filename

    cv2.imwrite(output_path, superimposed)  # save blended image
    print(f"Saved heatmap overlay to {output_path}")  # confirm save

def get_hotspot_bbox(heatmap, thresh=0.5):
    """
    Find bounding box of heatmap values > thresh.
    Returns (x, y, w, h) or None if no hotspot found.
    """
    mask = heatmap > thresh  # boolean mask of hotspots
    if not mask.any():
        return None  # no hotspot above threshold
    ys, xs = np.where(mask)  # get coords of mask
    x_min, x_max = xs.min(), xs.max()  # min/max in x
    y_min, y_max = ys.min(), ys.max()  # min/max in y
    return x_min, y_min, x_max - x_min, y_max - y_min  # return bbox

def occlude_box_on_hotspot(img, heatmap, thresh=0.5, color=(0,0,0)):
    """
    Draw a filled box over the hotspot region, correctly scaled to img size.
    """
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # upscale heatmap
    bbox = get_hotspot_bbox(heatmap_resized, thresh)  # compute hotspot bbox
    out = img.copy()  # copy for non-destructive edit
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness=-1)  # draw filled rectangle
    return out  # return occluded image

def occlude_blur_on_hotspot(img, heatmap, thresh=0.5, ksize=(51,51)):
    """
    Blur out the hotspot, scaled to img size.
    """
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # upscale heatmap
    bbox = get_hotspot_bbox(heatmap_resized, thresh)  # compute hotspot bbox
    out = img.copy()  # copy for non-destructive edit
    if bbox is not None:
        x, y, w, h = bbox
        roi = out[y:y+h, x:x+w]  # region of interest
        out[y:y+h, x:x+w] = cv2.GaussianBlur(roi, ksize, sigmaX=0)  # apply blur
    return out  # return occluded image

def occlude_pixelate_on_hotspot(img, heatmap, thresh=0.5, blocks=10):
    """
    Pixelate the hotspot, scaled to img size.
    """
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # upscale heatmap
    bbox = get_hotspot_bbox(heatmap_resized, thresh)  # compute hotspot bbox
    out = img.copy()  # copy for non-destructive edit
    if bbox is not None:
        x, y, w, h = bbox
        roi = out[y:y+h, x:x+w]  # region of interest
        small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)  # downsample
        out[y:y+h, x:x+w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)  # upsample as pixels
    return out  # return occluded image

def classify_image_with_gradcam(image_path):
    """
    Perform classification and then visualize Grad-CAM overlay.
    """
    img_array, decoded_predictions = classify_image(image_path)  # classify original image
    if img_array is None:
        return  # abort on error

    last_conv_layer = "Conv_1"  # final conv layer in MobileNetV2
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)  # compute heatmap
    overlay_heatmap(image_path, heatmap)  # save overlay

def classify_image_with_gradcam_and_occlusion(image_path):
    """
    1) Compute Grad-CAM (and save the overlay).
    2) Occlude hotspot three ways (saving each).
    3) Classify each occluded image and print its top-3.
    """
    # 1) Classify original & get tensor
    img_tensor, _ = classify_image(image_path)
    if img_tensor is None:
        return  # abort on error

    # 2) Compute & save Grad-CAM overlay
    heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_layer_name="Conv_1")
    overlay_heatmap(image_path, heatmap)

    # 3) Load original for occlusion
    orig = cv2.imread(image_path)  # read BGR image
    base, ext = os.path.splitext(image_path)  # split base name & extension
    occ_files = []  # list to hold generated filenames

    # Occlusion 1: black box
    fname1 = f"{base}-occ1{ext}"
    occ1 = occlude_box_on_hotspot(orig, heatmap, thresh=0.5, color=(0,0,0))
    cv2.imwrite(fname1, occ1)  # save file
    occ_files.append(fname1)  # track filename

    # Occlusion 2: blur
    fname2 = f"{base}-occ2{ext}"
    occ2 = occlude_blur_on_hotspot(orig, heatmap, thresh=0.5, ksize=(51,51))
    cv2.imwrite(fname2, occ2)  # save file
    occ_files.append(fname2)  # track filename

    # Occlusion 3: pixelate
    fname3 = f"{base}-occ3{ext}"
    occ3 = occlude_pixelate_on_hotspot(orig, heatmap, thresh=0.5, blocks=10)
    cv2.imwrite(fname3, occ3)  # save file
    occ_files.append(fname3)  # track filename

    # 4) Classify each occluded image and print its Top-3
    for i, f in enumerate(occ_files, start=1):
        print(f"\nTop-3 Predictions for occlusion #{i} ({os.path.basename(f)}):")
        classify_image(f)  # classify and print

#Keeping the original classify callouts to run a base script on a new pic or to just generate heatmap
#if __name__ == "__main__":
    #classify_image("avatar-256.jpg")
    #classify_image_with_gradcam("avatar-256.jpg")

classify_image_with_gradcam_and_occlusion("avatar-256.jpg")