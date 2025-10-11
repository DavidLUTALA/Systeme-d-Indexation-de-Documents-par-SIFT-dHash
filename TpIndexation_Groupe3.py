import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
import matplotlib.pyplot as plt

# === Fonction de hachage dHash ===
def compute_dhash(image, hash_size=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]

    dhash_value = 0
    for i, val in enumerate(diff.flatten()):
        if val:
            dhash_value |= 1 << (len(diff.flatten()) - 1 - i)

    return f"{dhash_value:016x}"

def compute_image_hash(image_bgr):
    return compute_dhash(image_bgr)

def extract_sift_descriptors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2)

def load_image_database(image_folder="images"):
    base_descripteurs = {}
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                keypoints, descriptors = extract_sift_descriptors(img)
                img_hash = compute_image_hash(img)
                base_descripteurs[filename] = {
                    "image": img,
                    "keypoints": keypoints,
                    "descriptors": descriptors,
                    "hash": img_hash
                }
    return base_descripteurs

def main():
    st.set_page_config(page_title="Indexation et Localisation de Texte", layout="wide")
    st.title("Système d'Indexation avec SIFT + dHash")

    st.sidebar.header("Téléversez une vidéo de requête")
    uploaded_video = st.sidebar.file_uploader("Vidéo .mp4", type=["mp4"])

    if uploaded_video:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_file.name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            st.error("Impossible d'extraire la frame de la vidéo.")
            return

        st.subheader("Frame extraite")
        kp_frame, des_frame = extract_sift_descriptors(frame)
        frame_hash = compute_image_hash(frame)

        img_kp = cv2.drawKeypoints(frame, kp_frame, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        st.image(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB), caption=f"{len(kp_frame)} keypoints détectés")
        st.code(f"Hash vidéo : {frame_hash}")

        st.info("Chargement de la base d'images...")
        base_descripteurs = load_image_database()

        hash_distances = {
            name: abs(int(data["hash"], 16) - int(frame_hash, 16))
            for name, data in base_descripteurs.items()
        }

        top_k = 30
        top_k_hashes = sorted(hash_distances.items(), key=lambda x: x[1])[:top_k]

        st.subheader("Top 30 par hachage dHash")
        for i, (name, dist) in enumerate(top_k_hashes, 1):
            st.write(f"{i:02d}. {name} – Distance : {dist} – Hash : {base_descripteurs[name]['hash']}")

        scores = {}
        for name, _ in top_k_hashes:
            des_img = base_descripteurs[name]["descriptors"]
            if des_img is not None and des_frame is not None:
                matches = bf.knnMatch(des_frame, des_img, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
                scores[name] = len(good_matches)

        best_match = max(scores, key=scores.get)
        nb_best_matches = scores[best_match]

        st.success(f"Document reconnu : {best_match} avec {nb_best_matches} bons matchs SIFT")
        best_img = base_descripteurs[best_match]["image"]
        kp_img = base_descripteurs[best_match]["keypoints"]
        des_img = base_descripteurs[best_match]["descriptors"]

        st.image(cv2.cvtColor(best_img, cv2.COLOR_BGR2RGB), caption="Image du document associé")

        matches = bf.knnMatch(des_frame, des_img, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) >= 10:
            src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            h_frame, w_frame = frame.shape[:2]
            rect = np.float32([[0, 0], [w_frame, 0], [w_frame, h_frame], [0, h_frame]]).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(rect, H)

            img_localised = best_img.copy()
            img_localised = cv2.polylines(img_localised, [np.int32(projected)], isClosed=True, color=(0, 255, 0), thickness=3)

            st.subheader("Zone localisée dans le document")
            st.image(cv2.cvtColor(img_localised, cv2.COLOR_BGR2RGB))
        else:
            st.warning("Pas assez de bons matchs pour estimer une homographie.")

if __name__ == "__main__":
    main()

