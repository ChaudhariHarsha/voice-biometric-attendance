# Full Stack Voice Biometric Attendance System (Offline)

# -----------------------------
# BACKEND + UI: Python (Streamlit + Resemblyzer)
# -----------------------------

# 1. requirements.txt
# -----------------------------
# streamlit
# numpy
# resemblyzer
# scipy
# sounddevice
# webrtcvad
# pyaudio

# Install with: pip install -r requirements.txt

# 2. streamlit_app.py (Run this file to start app)
# -----------------------------
import streamlit as st
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
from datetime import date
import os
import json
import time
import io
import csv

# Paths
DB_PATH = "data"
EMBED_DIR = os.path.join(DB_PATH, "embeddings")
ATTENDANCE_FILE = os.path.join(DB_PATH, "attendance.json")
STUDENTS_FILE = os.path.join(DB_PATH, "students.json")
os.makedirs(EMBED_DIR, exist_ok=True)
encoder = VoiceEncoder()

# Load or init students
if os.path.exists(STUDENTS_FILE):
    with open(STUDENTS_FILE, 'r') as f:
        students = json.load(f)
else:
    students = {}

def save_students():
    with open(STUDENTS_FILE, 'w') as f:
        json.dump(students, f)

def load_embedding(name):
    return np.load(os.path.join(EMBED_DIR, f"{name}.npy"))

def save_embedding(name, emb):
    np.save(os.path.join(EMBED_DIR, f"{name}.npy"), emb)

def mark_attendance(name):
    today = date.today().isoformat()
    if not os.path.exists(ATTENDANCE_FILE):
        attendance = {}
    else:
        with open(ATTENDANCE_FILE, 'r') as f:
            attendance = json.load(f)
    if today not in attendance:
        attendance[today] = []
    if name not in attendance[today]:
        attendance[today].append(name)
    with open(ATTENDANCE_FILE, 'w') as f:
        json.dump(attendance, f)

def get_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        return {}
    with open(ATTENDANCE_FILE, 'r') as f:
        return json.load(f)

def record_audio(duration=3, fs=16000):
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio, fs

st.set_page_config(page_title="Voice Attendance App")
st.title("🎙️ Voice Biometric Attendance")

menu = st.sidebar.selectbox("Menu", ["Enroll Student", "Recognize Voice", "Show Attendance", "Show Enrolled Students"])

if menu == "Enroll Student":
    name = st.text_input("Student Name")
    std = st.text_input("Standard")
    div = st.text_input("Division")
    year = st.text_input("Academic Year")
    roll_no = st.text_input("Roll No")
    emergency_contact = st.text_input("Emergency Contact Number")

    if st.button("Record & Enroll") and name:
        st.info("Recording...")
        audio, fs = record_audio()
        st.success("Recording complete")
        temp_file = f"temp_{name}.wav"
        wavfile.write(temp_file, fs, audio)
        wav = preprocess_wav(temp_file)
        emb = encoder.embed_utterance(wav)
        save_embedding(name, emb)
        students[name] = {
            "name": name,
            "std": std,
            "div": div,
            "year": year,
            "roll_no": roll_no,
            "emergency_contact": emergency_contact
        }
        save_students()
        os.remove(temp_file)
        st.success(f"{name} enrolled successfully")

elif menu == "Recognize Voice":
    if "recognizing" not in st.session_state:
        st.session_state.recognizing = False

    if not st.session_state.recognizing:
        if st.button("▶️ Start Recognizing"):
            st.session_state.recognizing = True
    else:
        if st.button("⏹ Stop Recognizing"):
            st.session_state.recognizing = False

    placeholder = st.empty()

    while st.session_state.recognizing:
        audio, fs = record_audio()
        temp_file = "temp_rec.wav"
        wavfile.write(temp_file, fs, audio)
        wav = preprocess_wav(temp_file)
        rec_emb = encoder.embed_utterance(wav)
        os.remove(temp_file)

        matched = False
        for name, data in students.items():
            ref_emb = load_embedding(name)
            sim = 1 - cosine(ref_emb, rec_emb)
            if sim > 0.75:
                mark_attendance(name)
                placeholder.success(f"✅ Recognized: {name} (similarity: {sim:.2f})")
                matched = True
                break
        if not matched:
            placeholder.warning("❌ No match found")

        time.sleep(1)

elif menu == "Show Attendance":
    data = get_attendance()
    if not data:
        st.info("No attendance recorded yet.")
    else:
        grouped = {}
        for date_key, names in data.items():
            for name in names:
                info = students.get(name)
                if info:
                    key = (info["year"], info["std"], info["div"], date_key)
                    grouped.setdefault(key, []).append(info)

        for (year, std, div, date_key), group in sorted(grouped.items()):
            st.markdown(f"### 📅 Date: {date_key} | Year: {year} | Std: {std} | Div: {div}")
            st.table([{"Name": s["name"], "Roll No": s["roll_no"], "Emergency Contact": s["emergency_contact"]} for s in group])

elif menu == "Show Enrolled Students":
    st.subheader("📋 Enrolled Students")
    search = st.text_input("Search by name or roll no")
    filtered_students = {k: v for k, v in students.items() if search.lower() in k.lower() or search.lower() in v['roll_no'].lower()} if search else students

    grouped = {}
    for _, info in filtered_students.items():
        key = (info["year"], info["std"], info["div"])
        grouped.setdefault(key, []).append(info)

    for (year, std, div), group in sorted(grouped.items()):
        st.markdown(f"### 📚 Year: {year} | Std: {std} | Div: {div}")
        st.table([{"Name": s["name"], "Roll No": s["roll_no"], "Emergency Contact": s["emergency_contact"]} for s in group])
        for info in group:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**Name**: {info['name']} | **Roll No**: {info['roll_no']} | **Emergency Contact**: {info['emergency_contact']}")
            with col2:
                if st.button("✏️ Edit", key=f"edit_{info['name']}"):
                    st.session_state.editing = info['name']
                if st.button("🗑 Delete", key=f"delete_{info['name']}"):
                    if info['name'] in students:
                        del students[info['name']]
                        save_students()
                        st.experimental_rerun()
        st.markdown("---")

    if st.button("⬇ Export as CSV"):
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(["Name", "Roll No", "Standard", "Division", "Year", "Emergency Contact"])
        for s in students.values():
            csv_writer.writerow([s['name'], s['roll_no'], s['std'], s['div'], s['year'], s['emergency_contact']])
        st.download_button("Download CSV", csv_buffer.getvalue(), "students.csv", "text/csv")

# -----------------------------
# TO RUN:
# > streamlit run streamlit_app.py
# -----------------------------
# Works completely offline.
