# Manar Vink 24025838, Anna Tan 24214464, Research Project
 
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2
from mtcnn import MTCNN
import os
import shutil
import glob
import time
 
 
class ImageGUI:
    # Base code for GUI taken from Lab03 starter guide
 
    def __init__(self, master):
        self.master = master
        self.master.title("Feature Detection")
 
        # Initialize detector
        self.detector = MTCNN()
 
        # Create a border for the GUI
        self.border = tk.Frame(self.master, borderwidth=2, relief="groove")
        self.border.pack(expand=True, padx=10, pady=10)
 
        # Image frame (top)
        self.images_frame = tk.Frame(self.border)
        self.images_frame.pack(side=tk.TOP)
 
        # Control frame (bottom)
        self.controls_frame = tk.Frame(self.border)
        self.controls_frame.pack(side=tk.TOP, padx=10, pady=5)
 
        self.left_frame = tk.Frame(self.images_frame)
        self.right_frame = tk.Frame(self.images_frame)
        self.left_frame.pack(
            side=tk.LEFT, padx=10, pady=10)
        self.right_frame.pack(
            side=tk.LEFT,  padx=10, pady=10)
 
        # label for input image panel
        tk.Label(self.left_frame, text="Input Image").pack()
        self.image_label = tk.Label(self.left_frame)
        self.image_label.pack()
 
        # label for processed image panel
        tk.Label(self.right_frame, text="Processed Image").pack()
        self.result_label = tk.Label(self.right_frame)
        self.result_label.pack()
 
        # single image button: open file picker, load one image, run full pipeline, display results
        self.single_image_button = tk.Button(
            self.controls_frame, text="Single Image", command=self.single_image)
        self.single_image_button.pack(side=tk.LEFT, padx=5, pady=5)
 
        # bulk processing button
        self.bulk_processing_button = tk.Button(
            self.controls_frame, text="Bulk Processing", command=self.bulk_processing)
        self.bulk_processing_button.pack(side=tk.LEFT, padx=5, pady=5)
 
        # status labels for processing time and face count
        self.time_label = tk.Label(
            self.controls_frame, text="Processing Time: --")
        self.time_label.pack(side=tk.LEFT, padx=5)
        self.faces_label = tk.Label(
            self.controls_frame, text="Faces Found: --")
        self.faces_label.pack(side=tk.LEFT, padx=5)
 
    def single_image(self):
        # open file picker, load image
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[
                                               ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if not file_path:
            return
        self.file_path = file_path
 
        start = time.time()
 
        # load chosen image using PIL and convert to numpy array
        self.original_image = Image.open(file_path)
        cv_image = cv2.cvtColor(
            np.array(self.original_image), cv2.COLOR_RGB2BGR)
 
        # Resize original image for display in the left panel
        width, height = self.original_image.size
        max_size = 300
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_width = int(width * (max_size / height))
            new_height = max_size
        display_image = self.original_image.resize((new_width, new_height))
 
        # Convert resized original image to Tkinter format and display it
        photo = ImageTk.PhotoImage(display_image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
 
        # run the shared processing pipeline and display result
        processed_img, faces, _ = self.process_image(cv_image)
 
        result_pil = Image.fromarray(
            cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        width, height = result_pil.size
        max_size = 300
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_width = int(width * (max_size / height))
            new_height = max_size
        result_pil = result_pil.resize((new_width, new_height))
 
        processed_photo = ImageTk.PhotoImage(result_pil)
        self.result_label.configure(image=processed_photo)
        self.result_label.image = processed_photo
 
        # label processing time and number of faces found in each image
        elapsed = time.time() - start
        self.time_label.configure(text=f"Processing Time: {elapsed:.2f}s")
        self.faces_label.configure(text=f"Faces Found: {len(faces)}")
 
    def process_image(self, cv_image):
        # detect faces, draw landmarks, paste aligned thumbnails into corners
        cv_image_clean = cv_image.copy()
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb_image)
 
        img_h, img_w = cv_image.shape[:2]
        corners = [
            (0, 0),
            (img_w - 125, 0),
            (0, img_h - 125),
            (img_w - 125, img_h - 125),
        ]
 
        # draw all bounding boxes and landmarks
        for face in faces[:4]:
            x, y, w, h = face['box']
            left_eye = face['keypoints']['left_eye']
            right_eye = face['keypoints']['right_eye']
            nose = face['keypoints']['nose']
 
            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(cv_image, right_eye, 4, (0, 255, 0), -1)
            cv2.circle(cv_image, left_eye, 4, (0, 0, 255), -1)
            cv2.circle(cv_image, nose, 4, (255, 0, 0), -1)
 
        # paste aligned thumbnails on top
        for i, face in enumerate(faces[:4]):
            aligned_face = self.similarity_transformation(face, cv_image_clean)
 
            cv2.circle(aligned_face, (40, 40), 3, (0, 0, 255), -1)
            cv2.circle(aligned_face, (85, 40), 3, (0, 255, 0), -1)
            cv2.circle(aligned_face, (63, 70), 3, (255, 0, 0), -1)
 
            cx, cy = corners[i]
            cv_image[cy:cy+125, cx:cx+125] = aligned_face
            cv2.rectangle(cv_image, (cx, cy),
                          (cx+125, cy+125), (255, 255, 255), 2)
 
        return cv_image, faces, cv_image_clean # return also the clean image for 125x125s
 
    def similarity_transformation(self, face_data, source_image):
        # crop -> align -> resize to produce a 125x125 aligned face thumbnail
 
        right_eye = face_data['keypoints']['right_eye']
        left_eye = face_data['keypoints']['left_eye']
 
        # crop, extract a padded region around the bounding box to prevent black borders
        x, y, w, h = face_data['box']
        img_h, img_w = source_image.shape[:2]
        pad = int(max(w, h) * 0.6)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)
        cropped = source_image[y1:y2, x1:x2]
 
        # align, do similarity transform using landmarks adjusted to crop coords
        src_pts = np.array([
            [left_eye[0] - x1,  left_eye[1] - y1],
            [right_eye[0] - x1, right_eye[1] - y1],
        ], dtype=np.float32)
        dst_pts = np.array([[40, 40], [85, 40]], dtype=np.float32)
        transformation_matrix, _ = cv2.estimateAffinePartial2D(
            src_pts, dst_pts)
 
        # resize, warpAffine with output size 125x125 to align and resizes in one step
        aligned_face = cv2.warpAffine(
            cropped, transformation_matrix, (125, 125))
        return aligned_face
 
    def bulk_processing(self):
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        if not folder_path:
            return
        self.file_path = folder_path

        # wipe and recreate the output folder (spec requires underscore, not space)
        processed_folder = os.path.join(folder_path, "Processed_Images")
        if os.path.exists(processed_folder):
            shutil.rmtree(processed_folder)
        os.makedirs(processed_folder)

        image_paths = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                    glob.glob(os.path.join(folder_path, "*.jpeg")) + \
                    glob.glob(os.path.join(folder_path, "*.png")) + \
                    glob.glob(os.path.join(folder_path, "*.bmp"))

        start = time.time()
        total_faces = 0
        face_counter = 0 # global face count across all images for filename

        for image_path in image_paths:
            img = cv2.imread(image_path)
            _, faces, clean_img = self.process_image(img) # 3 outputs, 
            total_faces += len(faces) # counts faces

            # save each cropped aligned face separately (no landmarks, just the clean 125x125)
            for face in faces[:4]:
                clean_face = self.similarity_transformation(face, clean_img) # changed from img for clean without landmarks (face crops)
                save_path = os.path.join(processed_folder, f"Identity_0_face_{face_counter}.jpg") # saves
                cv2.imwrite(save_path, clean_face)
                face_counter += 1

        elapsed = time.time() - start

        # update GUI labels — identity count is 0 for now until clustering is added
        self.time_label.configure(text=f"Processing Time: {elapsed:.2f}s")
        self.faces_label.configure(text=f"Total {len(image_paths)} images processed in {elapsed:.2f}s. {total_faces} faces detected corresponding to 0 unique identities.")
 
 
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGUI(root)
    root.mainloop()
