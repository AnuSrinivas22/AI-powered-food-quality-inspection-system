
import streamlit as st

from src.utils import load_image_bgr, to_rgb, draw_detections, compute_defect_score, quality_score
from src.detector_yolo import YoloDetector
from src.classifier import QualityClassifier


st.set_page_config(page_title='Food Quality Inspection', layout='wide')

st.title('Food Quality Inspection System')
st.write('Upload an image. The system detects objects, crops the top detection, estimates defects, and outputs a quality score.')

with st.sidebar:
    st.header('Settings')
    yolo_model = st.text_input('YOLO model name/path', value='yolov8n.pt')
    yolo_conf = st.slider('YOLO confidence', min_value=0.05, max_value=0.90, value=0.25, step=0.05)
    max_dets = st.slider('Max detections to display', min_value=1, max_value=10, value=5, step=1)

    st.markdown('---')
    st.subheader('Classifier')
    weights_path = st.text_input('Classifier weights path', value='models/quality_classifier.pt')

uploaded = st.file_uploader('Upload food image (JPG/PNG)', type=['jpg', 'jpeg', 'png'])

@st.cache_resource
def get_models(yolo_model_name: str, yolo_conf_val: float, weights: str):
    det = YoloDetector(model_name=yolo_model_name, conf=float(yolo_conf_val))
    clf = QualityClassifier(weights_path=weights)
    return det, clf

if uploaded is None:
    st.info('Upload an image to start.')
    st.stop()

img_bgr = load_image_bgr(uploaded.getvalue())

try:
    detector, classifier = get_models(yolo_model, yolo_conf, weights_path)
except Exception as e:
    st.error('Failed to load YOLO model. If this is the first run, it may be downloading weights. Try again in a minute.')
    st.exception(e)
    st.stop()

try:
    dets = detector.detect(img_bgr)
except Exception as e:
    st.error('Detection failed.')
    st.exception(e)
    st.stop()

if len(dets) == 0:
    st.warning('No detections found. Try a clearer image or lower YOLO confidence.')
    st.image(to_rgb(img_bgr), caption='Input image', use_container_width=True)
    st.stop()

dets = dets[:int(max_dets)]

best = dets[0]
x1, y1, x2, y2 = best.xyxy
roi = img_bgr[y1:y2, x1:x2].copy()

cls_probs = classifier.predict_proba(roi)
defects = compute_defect_score(roi)
q = quality_score(defects['defect_score_0_100'], cls_probs)

col1, col2 = st.columns(2)

with col1:
    st.subheader('Detections')
    vis = draw_detections(img_bgr, dets)
    st.image(to_rgb(vis), caption='Detected objects', use_container_width=True)

with col2:
    st.subheader('Top detection ROI')
    st.image(to_rgb(roi), caption='ROI used for scoring', use_container_width=True)

st.markdown('---')

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader('Classifier probabilities')
    st.json(cls_probs)
    if not classifier.is_trained:
        st.warning('Classifier weights not found. Train the classifier to get meaningful probabilities.')

with c2:
    st.subheader('Defect metrics')
    st.json(defects)

with c3:
    st.subheader('Final quality output')
    st.metric('Quality score', str(round(float(q['quality_score_0_100']), 1)) + ' / 100')
    st.write('Grade: ' + str(q['grade']))
    st.write('Decision: ' + str(q['decision']))

st.markdown('---')

st.subheader('Explainability')
st.write('Higher spot area and higher edge density increase defect score; higher fresh probability improves final score.')
