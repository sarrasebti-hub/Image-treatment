
import sys
import tempfile
import html

import streamlit as st

from pathlib import Path
from PIL import Image

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from generation.caption_generation import CaptionGenerationSystem


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hybrid RAG Image Captioning",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DARK MODE CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* GLOBAL BACKGROUND */

.stApp {
    background: #0B1120;
    color: #F1F5F9;
}

header[data-testid="stHeader"] {
    background: #0B1120;
}

[data-testid="stAppViewContainer"] {
    background: #0B1120;
}

[data-testid="stMain"] {
    background: #0B1120;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263449;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

/* MAIN TITLE */

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: #A78BFA;
    margin-bottom: 5px;
    letter-spacing: 0.5px;
}

.subtitle {
    text-align: center;
    font-size: 15px;
    color: #94A3B8;
    margin-bottom: 25px;
}

/* SECTION TITLES */

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #C4B5FD;
    margin-top: 15px;
    margin-bottom: 12px;
}

h1, h2, h3, h4 {
    color: #E2E8F0 !important;
}

/* RESULT BOX */

.result-box {
    background: linear-gradient(135deg, #1E293B, #172554);
    border: 1px solid #6366F1;
    border-left: 5px solid #8B5CF6;
    border-radius: 14px;
    padding: 24px;
    margin-top: 10px;
    margin-bottom: 15px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
}

.caption-text {
    font-size: 21px;
    font-weight: 600;
    color: #DDD6FE;
    line-height: 1.6;
}

/* METRICS */

[data-testid="stMetric"] {
    background: #1E293B;
    border: 1px solid #334155;
    padding: 16px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}

[data-testid="stMetricValue"] {
    color: #C4B5FD !important;
}

/* BUTTONS */

.stButton > button {
    background: linear-gradient(90deg, #7C3AED, #6366F1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-weight: 700;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #6D28D9, #4F46E5);
    color: white;
    border: 1px solid #A78BFA;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: #1E293B;
    border: 1px dashed #6366F1;
    border-radius: 12px;
    padding: 15px;
}

/* EXPANDERS */

[data-testid="stExpander"] {
    background: #172033;
    border: 1px solid #334155;
    border-radius: 10px;
}

/* INPUT TEXT */

.stTextInput input {
    background: #1E293B;
    color: white;
}

/* DIVIDERS */

hr {
    border-color: #334155 !important;
}

/* GENERAL TEXT */

p, label, li {
    color: #CBD5E1;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #64748B;
    font-size: 13px;
    padding: 20px;
    line-height: 1.8;
}

/* CAPTION BADGES */

.caption-badge {
    display: inline-block;
    background: #312E81;
    color: #C4B5FD;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD HYBRID RAG SYSTEM ONLY ONCE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_system():

    system = CaptionGenerationSystem()
    system.initialize()

    return system


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "<h2 style='color:#A78BFA;'>🖼️ Hybrid RAG</h2>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### ⚙️ System Components")

    components = [
        ("🔹", "BLIP", "Initial Caption"),
        ("🔹", "YOLO", "Object Detection"),
        ("🔹", "CLIP", "Image Embeddings"),
        ("🔹", "FAISS", "Dense Retrieval"),
        ("🔹", "BM25", "Sparse Retrieval"),
        ("🔹", "RRF", "Hybrid Retrieval"),
        ("🔹", "FLAN-T5", "Final Caption")
    ]

    for icon, name, description in components:
        st.markdown(
            f"""
            <div style="
                background:#1E293B;
                padding:10px;
                border-radius:8px;
                margin-bottom:7px;
                border:1px solid #334155;
            ">
                <span style="color:#A78BFA;font-weight:bold;">
                    {icon} {name}
                </span>
                <br>
                <span style="color:#94A3B8;font-size:12px;">
                    {description}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### ℹ️ About")

    st.write(
        "This application generates image captions "
        "using a Hybrid Retrieval-Augmented Generation system."
    )

    st.markdown("---")

    st.caption("💻 CPU-based execution")
    st.caption("🎓 Academic Project")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🖼️ Hybrid RAG Image Captioning</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Image Description using Retrieval-Augmented Generation'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📤 Upload Your Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image from your computer",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# IMAGE PREVIEW
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1.2, 1])

    with col1:

        st.markdown("### 📷 Input Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.markdown("### 📋 Image Information")

        st.markdown(
            f"""
            <div style="
                background:#1E293B;
                border:1px solid #334155;
                border-radius:12px;
                padding:20px;
                line-height:2.2;
            ">
                <b style="color:#A78BFA;">Filename:</b>
                {html.escape(uploaded_file.name)}<br>

                <b style="color:#A78BFA;">Width:</b>
                {image.width} pixels<br>

                <b style="color:#A78BFA;">Height:</b>
                {image.height} pixels<br>

                <b style="color:#A78BFA;">Format:</b>
                {html.escape(uploaded_file.type or "Unknown")}<br>

                <b style="color:#A78BFA;">File size:</b>
                {uploaded_file.size / 1024:.2f} KB
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # ========================================================
    # GENERATE BUTTON
    # ========================================================

    if st.button(
        "🚀 Generate Caption",
        type="primary",
        use_container_width=True
    ):

        temp_image_path = None
        total_captions = 0

        try:

            with st.spinner(
                "Loading AI models and retrieval indexes..."
            ):

                system = load_system()

            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded_file.name).suffix,
                delete=False
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temp_image_path = temp_file.name

            with st.spinner(
                "Running Hybrid RAG pipeline..."
            ):

                result = system.generate(temp_image_path)

            st.success("Image caption generated successfully!")

            st.divider()

            # =================================================
            # RESULTS DASHBOARD
            # =================================================

            st.markdown("## 📊 Generation Results")

            # -------------------------------------------------
            # 1. BLIP
            # -------------------------------------------------

            st.markdown("### 1. Initial Caption — BLIP")

            st.info(result["initial_caption"])

            # -------------------------------------------------
            # 2. YOLO
            # -------------------------------------------------

            st.markdown("### 2. Object Detection — YOLO")

            objects = result["detected_objects"]

            if objects:

                st.write("Detected objects:")

                st.write(
                    " | ".join(
                        f"`{obj}`" for obj in objects
                    )
                )

                st.metric(
                    "Number of detected objects",
                    len(objects)
                )

            else:

                st.warning(
                    "No objects were detected in this image."
                )

            st.divider()

            # -------------------------------------------------
            # 3. HYBRID RETRIEVAL
            # -------------------------------------------------

            st.markdown("### 3. Hybrid Retrieval")

            retrieved_images = result["retrieved_images"]

            if retrieved_images:

                st.write(
                    f"Top {len(retrieved_images)} similar images:"
                )

                for index, item in enumerate(
                    retrieved_images,
                    start=1
                ):

                    with st.expander(
                        f"#{index} — {item['image_name']} "
                        f"| RRF: {item['rrf_score']:.6f}"
                    ):

                        st.write(
                            f"**Image ID:** {item['image_id']}"
                        )

                        st.write(
                            f"**RRF Score:** "
                            f"{item['rrf_score']:.6f}"
                        )

            else:

                st.warning(
                    "No similar images were retrieved."
                )

            st.divider()

            # -------------------------------------------------
            # 4. CAPTION RETRIEVAL
            # -------------------------------------------------

            st.markdown("### 4. Retrieved Captions")

            retrieved_captions = result["retrieved_captions"]

            if retrieved_captions:

                total_captions = sum(
                    len(item.get("captions", []))
                    for item in retrieved_captions
                )

                st.metric(
                    "Total retrieved captions",
                    total_captions
                )

                for index, item in enumerate(
                    retrieved_captions,
                    start=1
                ):

                    captions = item.get("captions", [])

                    with st.expander(
                        f"Image {index}: {item['image_name']} "
                        f"({len(captions)} captions)",
                        expanded=(index == 1)
                    ):

                        st.write(
                            f"**Image ID:** {item['image_id']}"
                        )

                        st.write(
                            f"**RRF Score:** "
                            f"{item['rrf_score']:.6f}"
                        )

                        st.markdown("**Captions:**")

                        if captions:

                            for caption_index, caption in enumerate(
                                captions,
                                start=1
                            ):

                                st.markdown(
                                    f"**{caption_index}.** {caption}"
                                )

                        else:

                            st.warning(
                                "No captions available for this image."
                            )

            else:

                st.warning(
                    "No captions were returned by Caption Retrieval."
                )

            st.divider()

            # -------------------------------------------------
            # 5. FINAL CAPTION
            # -------------------------------------------------

            st.markdown("## ✨ 5. Final Caption — FLAN-T5")

            final_caption = html.escape(
                result["final_caption"]
            )

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="caption-text">
                        {final_caption}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # -------------------------------------------------
            # FALLBACK STATUS
            # -------------------------------------------------

            st.markdown("### Caption Validation")

            if result["used_fallback"]:

                st.warning(
                    "Fallback activated: the FLAN-T5 output "
                    "did not pass validation. The initial BLIP "
                    "caption was used instead."
                )

            else:

                st.success(
                    "Validation passed: FLAN-T5 output was accepted."
                )

            # -------------------------------------------------
            # FINAL SUMMARY
            # -------------------------------------------------

            st.divider()

            st.markdown("### 📌 Pipeline Summary")

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Detected Objects",
                    len(objects)
                )

            with c2:

                st.metric(
                    "Retrieved Images",
                    len(retrieved_images)
                )

            with c3:

                st.metric(
                    "Retrieved Captions",
                    total_captions
                )

        except Exception as error:

            st.error(f"An error occurred: {error}")

            with st.expander("Technical details"):

                st.exception(error)

        finally:

            if temp_image_path:

                Path(temp_image_path).unlink(
                    missing_ok=True
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <b>Hybrid RAG Image Captioning Project</b><br>
        BLIP • YOLO • CLIP • FAISS • BM25 • RRF • FLAN-T5
    </div>
    """,
    unsafe_allow_html=True
)
