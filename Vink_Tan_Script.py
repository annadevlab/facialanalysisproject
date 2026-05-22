# Manar Vink 24025838, Anna Tan 24214464, Research Project

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2
from mtcnn import MTCNN
import os
import glob
import time


class ImageGUI:
    # Base code for GUI taken from Lab03 starter guide

    def __init__(self, master):
        # make GUI and initialise all frames and buttons

        self.master = master
        self.master.title("Feature Detection")

        # initialise MTCNN feature detector
        self.detector = MTCNN()

        # load SFace model
        # face_recognition_sface_2021dec.onnx must be in the same folder as this script
        _model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "face_recognition_sface_2021dec.onnx")
        self.face_recognizer = cv2.FaceRecognizerSF.create(_model_path, "")

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

        # bulk processing button: open folder picker, process all images, cluster identities
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
        # choose image to process, resize final img and display

        # open file picker, load image
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[
                                               ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if not file_path:
            return
        self.file_path = file_path

        # start timer for processing time
        start = time.time()

        # load chosen image using PIL and convert to numpy array
        self.original_image = Image.open(file_path)
        cv_image = cv2.cvtColor(
            np.array(self.original_image), cv2.COLOR_RGB2BGR)

        # resize original image for display in the left panel
        width, height = self.original_image.size
        max_size = 300
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_width = int(width * (max_size / height))
            new_height = max_size
        display_image = self.original_image.resize((new_width, new_height))

        # convert resized original image to Tkinter format and display it
        photo = ImageTk.PhotoImage(display_image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

        # run the processing pipeline and display result
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

        # display processed photo
        processed_photo = ImageTk.PhotoImage(result_pil)
        self.result_label.configure(image=processed_photo)
        self.result_label.image = processed_photo

        # label processing time and number of faces found in each image
        elapsed = time.time() - start
        self.time_label.configure(text=f"Processing Time: {elapsed:.2f}s")
        self.faces_label.configure(text=f"Faces Found: {len(faces)}")

    def process_image(self, cv_image):
        # detect faces, draw landmarks, paste aligned thumbnails into corners

        # store clean image for unmarked pixels for face crops
        cv_image_clean = cv_image.copy()
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb_image)

        # filter for confidence then use skin filter to remove remaining false positives
        faces = [f for f in faces if f['confidence'] > 0.95]
        faces = self.skin_filter(cv_image, faces)

        # define corner positions for thumbnail placement
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

            # landmark dots at spec target positions
            cv2.circle(aligned_face, (40, 40), 3, (0, 0, 255), -1)   # left eye
            cv2.circle(aligned_face, (85, 40), 3,
                       (0, 255, 0), -1)   # right eye
            cv2.circle(aligned_face, (63, 70), 3, (255, 0, 0), -1)   # nose

            cx, cy = corners[i]
            cv_image[cy:cy+125, cx:cx+125] = aligned_face
            cv2.rectangle(cv_image, (cx, cy),
                          (cx+125, cy+125), (255, 255, 255), 2)

        # return also the clean image for 125x125s
        return cv_image, faces, cv_image_clean

    def similarity_transformation(self, face_data, source_image):
        # crop -> align -> resize to 125x125

        left_eye = face_data['keypoints']['left_eye']
        right_eye = face_data['keypoints']['right_eye']

        # crop with padding to prevent black borders after warpAffine rotates crop
        x, y, w, h = face_data['box']
        img_h, img_w = source_image.shape[:2]
        pad = int(max(w, h) * 0.6)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)
        cropped = source_image[y1:y2, x1:x2]

        # 2-point similarity transform: exactly maps both eyes to their target positions
        src_pts = np.array([
            [left_eye[0] - x1, left_eye[1] - y1],
            [right_eye[0] - x1, right_eye[1] - y1],
        ], dtype=np.float32)
        dst_pts = np.array([
            [40.0, 40.0],   # left eye
            [85.0, 40.0],   # right eye
        ], dtype=np.float32)
        transformation_matrix, _ = cv2.estimateAffinePartial2D(
            src_pts, dst_pts)
        aligned_face = cv2.warpAffine(
            cropped, transformation_matrix, (125, 125))
        return aligned_face

    def skin_filter(self, cv_image, faces):
        # return filtered list of faces (contains enough skin-coloured pixels)

        img_h, img_w = cv_image.shape[:2]
        filtered = []

        for face in faces:
            x, y, w, h = face['box']

            # clamp to image box bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x+w), min(img_h, y+h)
            roi = cv_image[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # convert to YCrCb and threshold for skintones (separate brightness from colour)
            ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

            # threshold Cr and Cb for range of skin tones
            skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

            # count what fraction of pixels pass threshold
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size

            if skin_ratio >= 0.15:
                filtered.append(face)

        return filtered

    def extract_embedding(self, face_img):
        # SFace needs 112x112 BGR, resize 125x125 aligned thumbnail
        face_112 = cv2.resize(face_img, (112, 112))
        return self.face_recognizer.feature(face_112).flatten()

    def cluster_identities(self, embeddings):
        # scale embedding vector to length 1, gives cosine similarity between faces
        arr = np.array(embeddings, dtype=np.float32)
        arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)  # L2 normalise
        n = len(arr)
        if n == 1:
            return [0]

        # build adjacency matrix, tune COSINE_THRESHOLD
        sim = arr @ arr.T
        COSINE_THRESHOLD = 0.48
        adj = sim > COSINE_THRESHOLD
        np.fill_diagonal(adj, False)

        # BFS connected components, each component is one identity
        labels = [-1] * n
        label = 0
        for start in range(n):
            if labels[start] != -1:
                continue
            queue = [start]
            labels[start] = label
            while queue:
                node = queue.pop()
                for nb in range(n):
                    if labels[nb] == -1 and adj[node, nb]:
                        labels[nb] = label
                        queue.append(nb)
            label += 1
        return labels

    def bulk_processing(self):
        # choose folder, process images, create a folder to store identities

        # open folder picker
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        if not folder_path:
            return
        self.file_path = folder_path

        # create Processed_Images subfolder (or clean it), collect all valid image paths via glob
        processed_folder = os.path.join(folder_path, "Processed_Images")
        if os.path.exists(processed_folder):
            for f in glob.glob(os.path.join(processed_folder, "*.jpg")) + \
                    glob.glob(os.path.join(processed_folder, "*.png")) + \
                    glob.glob(os.path.join(processed_folder, "*.bmp")):
                os.remove(f)
        else:
            os.makedirs(processed_folder)

        image_paths = glob.glob(os.path.join(folder_path, "*.jpg")) + \
            glob.glob(os.path.join(folder_path, "*.jpeg")) + \
            glob.glob(os.path.join(folder_path, "*.png")) + \
            glob.glob(os.path.join(folder_path, "*.bmp"))

        # label to show currently processing
        self.faces_label.configure(
            text=f"Processing {len(image_paths)} images...")
        self.master.update()

        # start timer for processing time readout
        start = time.time()
        total_faces = 0
        all_faces = []
        all_embeddings = []

        # for each image run process_image, similarity_transformation again to get saved crops
        for image_path in image_paths:
            img = cv2.imread(image_path)
            _, faces, clean_img = self.process_image(img)
            total_faces += len(faces)

            for face in faces[:4]:
                clean_face = self.similarity_transformation(face, clean_img)
                all_faces.append(clean_face)
                all_embeddings.append(
                    self.extract_embedding(clean_face))

        # send embeddings to cluster_identities and save thumbnails as labelled identities
        labels = self.cluster_identities(all_embeddings)
        n_identities = len(set(labels))

        # save each face crop under its identity label
        identity_counters = {}
        for face_img, label in zip(all_faces, labels):
            count = identity_counters.get(label, 0)
            save_path = os.path.join(
                processed_folder, f"Identity_{label}_face_{count}.jpg")
            cv2.imwrite(save_path, face_img)
            identity_counters[label] = count + 1

        elapsed = time.time() - start

        # label for processing time and number of identities
        self.time_label.configure(text=f"Processing Time: {elapsed:.2f}s")
        self.faces_label.configure(
            text=f"Total {len(image_paths)} images processed in {elapsed:.2f}s. "
            f"{total_faces} faces detected corresponding to {n_identities} unique identities.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGUI(root)
    root.mainloop()
