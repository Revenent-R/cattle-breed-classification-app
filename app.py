import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from json import load
from streamlit_autorefresh import st_autorefresh
from time import sleep

st.set_page_config(layout="wide", page_title="Cattle Breed Classifier", page_icon="")


@st.cache_resource
def load_model():
    return YOLO("best_19.pt")


@st.cache_resource
def load_model_detect():
    return YOLO("best_latest_detect.pt")


@st.cache_resource
def load_json():
    with open("data/cattle_info_updated.json", "r", encoding="utf-8") as f:
        return load(f)


if "run" not in st.session_state:
    st.session_state.run = None
if "box" not in st.session_state:
    st.session_state.box = None
if "frame" not in st.session_state:
    st.session_state.frame = None

st.markdown("""
    <style>
    body {
        color: #fafafa;
        #background-color: #0E1117;
    }
    div[data-testid="stAppViewContainer"] {
        #background-color: #0E1117;
    }
    div[data-testid="stSidebar"] {
        background-color: #262730;
    }
    .stButton>button {
        color: #262730;
        background-color: #FFFFFF;
        border: 1px solid #444;
    }
    .stProgress > div > div > div > div {
        background-color: #32cd32; 
    }
    </style>
    """, unsafe_allow_html=True)

model = load_model()
detector = load_model_detect()
js_data = load_json()


def display_results(predicted_class, confidence_score):
    with right:
        st.success(f"**Predicted Breed: {predicted_class}**")

        if confidence_score < 0.55:
            color = "#dc143c"
        elif confidence_score < 0.80:
            color = "#ffbf00"
        else:
            color = "#32cd32"

        st.markdown(f"""
            <style>
                .stProgress > div > div > div > div {{
                    background-color: {color} !important;
                }}
            </style>
        """, unsafe_allow_html=True)
        st.subheader("Confidence Score")
        st.progress(confidence_score, text=f"{confidence_score:.2%}")

        lang_list = ["English", "Hindi", "Marathi", "Tamil", "Telugu", "Malyalam", "Kannada"]
        option = st.selectbox("Select Language", lang_list)
        language = js_data.get(option)
        st_autorefresh(100, limit=1)
        breed_info = language.get(predicted_class)

        tab_titles = ["Overview", "Husbandry", "Production", "Health"]
        tabs = st.tabs(tab_titles)

        info_map = {
            "Overview": ['Name', 'Origin', 'Description', 'Uses'],
            "Husbandry": ['Region', 'Environment', 'water_intake', 'seasonal_diet'],
            "Production": ['reproduction_and_lactation', 'milk_yield', 'special_feeding_during_lactation'],
            "Health": ['health_management', 'deworming_frequency', 'common_diseases_and_symptoms', 'first_aid_tips']
        }

        for i, tab in enumerate(tabs):
            with tab:
                keys_for_tab = info_map[tab_titles[i]]
                for key in keys_for_tab:
                    title = key.replace('_', ' ').title()
                    content = breed_info.get(key.lower(), "N/A")
                    with st.expander(title):
                        st.write(content)


def crop_nearest(tag, img, coords):
    area = []
    for box in coords:
        x = box[2] - box[0]
        y = box[3] - box[1]
        area.append(x * y)
    initial = img[area.index(max(area))].shape
    init_coords = [0, 0, initial[1], initial[0]]
    c_img = Image.fromarray(img[area.index(max(area))])
    width, height = c_img.size
    f_coords = coords[area.index(max(area))]
    crop_resized = None
    for i in range(20):
        int_factor = i / 20
        x1 = ((1 - int_factor) * init_coords[0]) + (int_factor * f_coords[0])
        y1 = ((1 - int_factor) * init_coords[1]) + (int_factor * f_coords[1])
        x2 = ((1 - int_factor) * init_coords[2]) + (int_factor * f_coords[2])
        y2 = ((1 - int_factor) * init_coords[3]) + (int_factor * f_coords[3])
        crop = c_img.crop((int(x1), int(y1), int(x2), int(y2)))
        crop_resized = crop.resize((width, height), Image.LANCZOS)
        sleep(0.05)
        tag.image(crop_resized)
    upscaled = cv2.resize(np.array(crop_resized), None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    tag.image(upscaled, caption="Uploaded Image", use_container_width=True)

    return c_img


st.title("Cattle & Buffalo Breed Classifier")

mode = st.sidebar.radio("Choose Your Mode", ["Upload Image", "Live Camera"])

left, right = st.columns([1, 1.2])

if mode == "Upload Image":
    with left:
        st.header("Upload an Image")
        file = st.file_uploader("Select a file", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

        if file:
            pil_image = Image.open(file)
            box_coords = []
            img_s = []
            detect_result = detector.predict(pil_image)
            img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            for result in detect_result:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    box_coords.append((x1, y1, x2, y2))
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    img_s.append(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            image = st.empty()
            if len(img_s) > 0:
                f_img = crop_nearest(image, img_s, box_coords)
                results = model.predict(f_img)

                if results and results[0].probs:
                    top_class = results[0].names[results[0].probs.top1]
                    top_confidence = float(results[0].probs.top1conf)
                    display_results(top_class, top_confidence)
            else:
                st.error("No cattle were detected in the image uploaded")
        else:
            with right:
                st.info("Please upload an image to see the classification results here.")

elif mode == "Live Camera":
    with left:
        st.header("Live Camera Feed")
        run = st.checkbox("Start Camera", key="camera_run")

        frame_placeholder = st.empty()

        if not run:
            st.session_state.frame = None

        if st.session_state.frame is not None:
            frame_placeholder.image(st.session_state.frame, caption="Captured Image")
            results = model.predict(st.session_state.frame)
            if results and results[0].probs:
                top_class = results[0].names[results[0].probs.top1]
                top_confidence = float(results[0].probs.top1conf)
                display_results(top_class, top_confidence)
                st.session_state.run = None
        else:
            if run:
                st.session_state.frame = None
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not open webcam.")
                else:
                    st.session_state.run = run
                    if st.button("Capture & Identify", key="camera_capture"):
                        ret, frame = cap.read()
                        if ret:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            x1, y1, x2, y2 = st.session_state.box
                            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                            pil_img = Image.fromarray(rgb_frame)
                            pil_img = pil_img.crop((x1, y1, x2, y2))
                            frame_placeholder.image(pil_img, caption="Captured Image")

                            results = model.predict(pil_img)
                            if results and results[0].probs:
                                top_class = results[0].names[results[0].probs.top1]
                                top_confidence = float(results[0].probs.top1conf)
                                display_results(top_class, top_confidence)
                                st.session_state.run = None
                            st.session_state.frame = pil_img
                        else:
                            st.error("Failed to capture a frame.")
                    while run:
                        if 'run' in st.session_state and st.session_state.run:
                            ret, frame = cap.read()
                            if ret:
                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                results = detector.predict(frame, conf=0.4)
                                for r in results:
                                    for box in r.boxes:
                                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                        st.session_state.box = (x1, y1, x2, y2)
                                        cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                frame_placeholder.image(rgb_frame, channels="RGB")

                cap.release()
            else:
                with right:
                    st.info("Start the camera feed and click 'Capture & Identify' to see results.")