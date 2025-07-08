import os, cv2, imutils
from tqdm import tqdm

def crop_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeft, extRight = tuple(c[c[:, :, 0].argmin()][0]), tuple(c[c[:, :, 0].argmax()][0])
    extTop, extBot = tuple(c[c[:, :, 1].argmin()][0]), tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    return img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

def process_images(root_dir, save_dir, img_size=(256, 256)):
    for subdir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, subdir)
        save_path = os.path.join(save_dir, subdir)
        os.makedirs(save_path, exist_ok=True)
        for fname in tqdm(os.listdir(class_path), desc=subdir):
            img_path = os.path.join(class_path, fname)
            image = cv2.imread(img_path)
            cropped = crop_img(image)
            resized = cv2.resize(cropped, img_size)
            cv2.imwrite(os.path.join(save_path, fname), resized)
