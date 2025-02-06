import streamlit as st
from PIL import Image
import numpy as np
import cv2
import sqlite3
import pytesseract
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("your_model.pt")

def detect_vehicle_model_and_color(image_path):
    """Call the existing Gemini API script to detect vehicle model and color."""
    try:
        # Run the Gemini script and pass the image path as an argument
        result = subprocess.run(["python", "check_color1.py", image_path], 
                                capture_output=True, text=True)
        output = result.stdout.strip()

        # Parse the JSON output from the script
        data = json.loads(output)
        vehicle_make = data.get("make", "Unknown Make")
        vehicle_model = data.get("model", "Unknown Model")
        vehicle_color = data.get("color", "Unknown Color")
        return vehicle_make,vehicle_model, vehicle_color
    except Exception as e:
        print(f"Error running Gemini script: {e}")
        return "Unknown Model", "Unknown Color","Unknown Make"

def process_image(image):
    # Convert PIL image to OpenCV format
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run detection using the trained YOLOv8 model
    results = model(image_cv)

    # Placeholder: Extract detected plate number
    plate_text = pytesseract.image_to_string(image_cv, config='--psm 8').strip()

    # Check against database
    match = check_database(plate_text)

    return plate_text, vehicle_model, vehicle_color, match

def check_database(plate_text):
    conn = sqlite3.connect("vehicles.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vehicle_data WHERE plate_number = ?", (plate_text,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

st.title("Vehicle Recognition System")
st.write("Upload an image to detect the vehicle's license plate, model, and color.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    plate, model, color, match = process_image(image)

    st.subheader("Detection Results")
    st.write(f"**License Plate:** {plate}")
    st.write(f"**Vehicle Model:** {model}")
    st.write(f"**Vehicle Color:** {color}")

    if match:
        st.success("✅ Vehicle matches database records.")
    else:
        st.error("❌ Mismatch detected! Potential fake plate.")
