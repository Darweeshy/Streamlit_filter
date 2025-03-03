
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.title('Image Filters Application')
# Upload image
uploaded_image = st.file_uploader('Choose an Image', type=['jpg', 'png', 'jpeg'])

def black_white(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_image

def blur_image(img, ksize=5):
    blur = cv2.GaussianBlur(img, (ksize, ksize), 0, 0)
    sketch, _ = cv2.pencilSketch(blur)  
    return sketch

def vintage(img, level=2):
    height, width = img.shape[:2]
    x_kernel = cv2.getGaussianKernel(width, width/level)
    y_kernel = cv2.getGaussianKernel(height, height/level)
    kernel = y_kernel * x_kernel.T
    mask = kernel / kernel.max()
    image_copy = np.copy(img)
    for i in range(3):
        image_copy[:, :, i] = image_copy[:, :, i] * mask
    return image_copy
    
def remove_hair(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted_image = cv2.inpaint(img, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
    inpainted_image = cv2.cvtColor(inpainted_image , cv2.COLOR_BGR2RGB)
    return inpainted_image




def HDR(img, level, sigma_s=10, sigma_r=0.1):
    bright = cv2.convertScaleAbs(img, beta=level)
    hd_image = cv2.detailEnhance(bright, sigma_s=sigma_s, sigma_r=sigma_r)
    return hd_image

def style_image(img, sigma_s=10, sigma_r=0.1):
    blur = cv2.GaussianBlur(img, (5,5), 0, 0)
    style = cv2.stylization(blur, sigma_s=sigma_s, sigma_r=sigma_r)  
    return style

def brightness(img, level):
    bright = cv2.convertScaleAbs(img, beta=level)
    return bright

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  

    original_image, output_image = st.columns(2)

    with original_image:
        st.header('Original')
        st.image(img, channels='BGR', use_column_width=True)
        st.header('Filters List')

    options = st.selectbox('Select Filters:', ('None', 'Gray Image', 'Blur Image', 'HDR', 'Vintage', 'Style Image', 'Brighten Image' , 'Remove Hair'))
    
    output = img  # Default output
    color = 'BGR'

    if options == 'None':
        output = img

    elif options == 'Gray Image':
        output = black_white(img)
        color = 'GRAY'  # Set color flag to grayscale

    elif options == 'Blur Image':
        kvalue = st.slider('Kernel Size', 1, 15, 5, 2)
        output = blur_image(img, kvalue)

    elif options == 'HDR':
        level = st.slider('HDR Level', 0, 100, 50, 10)
        output = HDR(img, level)

    elif options == 'Vintage':
        level = st.slider('Vintage Level', 1, 5, 2)
        output = vintage(img, level)

    elif options == 'Style Image':
        sigma_s = st.slider('Sigma S', 1, 100, 50)
        sigma_r = st.slider('Sigma R', 0.1, 1.0, 0.5)
        output = style_image(img, sigma_s, sigma_r)

    elif options == 'Brighten Image':
        level = st.slider('Brightness Level', 0, 100, 50, 10)
        output = brightness(img, level)
        
    elif options == 'Remove Hair':
        output = remove_hair(img)
        color = 'GRAY' 

    

    with output_image:
        st.header('Output Image')
        if color == 'BGR':
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB) 
            st.image(output, use_column_width=True)
        elif color == 'GRAY':
            st.image(output, use_column_width=True, channels='GRAY')  
