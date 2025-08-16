# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import smtplib
import hashlib
import hmac
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# ---------- CONFIG ----------
USERS_CSV = "users.csv"
MODEL_FILE = "health_model.pkl"   # ensure this exists in same folder
HISTORY_PREFIX = "history_"       # history_<username>.csv

# ---------- Helper functions ----------
def load_model(path=MODEL_FILE):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}. Put your trained model file there.")
        st.stop()
    return joblib.load(path)
    
# Function to create PDF report
# --------- PDF (line-by-line) ----------
def generate_pdf_report_line_by_line(df, patient_name, username):
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    def draw_header(c, width, height):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Patient Health Report")
        c.setFont("Helvetica", 11)
        c.drawString(50, height - 75, f"Patient Name: {patient_name}")
        c.drawString(50, height - 90, f"Username: {username}")
        c.drawString(50, height - 105, f"Report Generated On: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Order + labels to print
    fields = [
        ("Heart Rate", "Heart Rate"),
        ("Respiratory Rate", "Respiratory Rate"),
        ("Body Temperature", "Body Temperature"),
        ("Oxygen Saturation", "Oxygen Saturation"),
        ("Systolic Blood Pressure", "Systolic Blood Pressure"),
        ("Diastolic Blood Pressure", "Diastolic Blood Pressure"),
        ("Age", "Age"),
        ("Gender", "Gender"),
        ("Derived BMI", "Derived_BMI"),
        ("Derived MAP", "Derived_MAP"),
        ("Risk", "Risk"),
    ]

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    c.setTitle(f"report_{username}")

    draw_header(c, width, height)

    y = height - 140
    line_h = 14

    # Ensure Date is printable string
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    for _, row in df.iterrows():
        # Page break if needed (leave bottom margin)
        if y < 90:
            c.showPage()
            draw_header(c, width, height)
            y = height - 70

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Date: {row['Date']}")
        y -= line_h

        c.setFont("Helvetica", 11)
        for label, col in fields:
            if col in row.index:
                val = row[col]
                # Make everything a clean string
                try:
                    if isinstance(val, float):
                        # keep integers without .0, otherwise 1–2 decimals
                        val = int(val) if float(val).is_integer() else round(float(val), 2)
                except Exception:
                    pass
                c.drawString(70, y, f"{label}: {val}")
                y -= line_h

        y -= 6  # small gap between records

    c.save()
    buf.seek(0)
    return buf
# Password hashing (PBKDF2-HMAC-SHA256)
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return salt.hex() + ":" + key.hex()

def verify_password(stored: str, provided: str) -> bool:
    try:
        salt_hex, key_hex = stored.split(":")
        salt = bytes.fromhex(salt_hex)
        stored_key = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac("sha256", provided.encode("utf-8"), salt, 100000)
        return hmac.compare_digest(new_key, stored_key)
    except Exception:
        return False

# Users CSV helpers
def ensure_users_csv():
    if not os.path.exists(USERS_CSV):
        df = pd.DataFrame(columns=["username", "name", "password", "caretaker_email"])
        df.to_csv(USERS_CSV, index=False)

def read_users():
    ensure_users_csv()
    return pd.read_csv(USERS_CSV)

def add_user(username, name, hashed_password, caretaker_email):
    df = read_users()
    if username in df["username"].values:
        return False, "Username already exists"
    df = pd.concat([df, pd.DataFrame([[username, name, hashed_password, caretaker_email]],
                                     columns=df.columns)], ignore_index=True)
    df.to_csv(USERS_CSV, index=False)
    return True, "User added"

# History helpers
def save_history(username, row_dict):
    filename = f"{HISTORY_PREFIX}{username}.csv"
    if os.path.exists(filename):
        hdf = pd.read_csv(filename)
    else:
        hdf = pd.DataFrame()
    hdf = pd.concat([hdf, pd.DataFrame([row_dict])], ignore_index=True)
    hdf.to_csv(filename, index=False)

def load_history(username):
    filename = f"{HISTORY_PREFIX}{username}.csv"
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return None

# Email alert
def get_email_credentials():
    # Try environment variables first
    sender = os.getenv("EMAIL_SENDER")
    app_pw = os.getenv("EMAIL_PASSWORD")
    if sender and app_pw:
        return sender, app_pw
    # Fallback for demo (⚠ for real use, set env vars instead)
    fallback_sender = "jadhavdhanashri0602@gmail.com"
    fallback_pw = "qoxg peuy nkwo crcx"  # Gmail App Password
    return fallback_sender, fallback_pw

def send_email_alert(patient_data: dict, receiver_email: str):
    sender_email, sender_password = get_email_credentials()
    if not sender_email or not sender_password:
        st.error("Email credentials not configured. Set EMAIL_SENDER and EMAIL_PASSWORD env vars or fill fallback in code.")
        return False

    subject = "🚨 High Risk Patient Alert"
    body_lines = ["ALERT: High Risk detected", "", "Patient details:"]
    for k, v in patient_data.items():
        body_lines.append(f"{k}: {v}")
    body = "\n".join(body_lines)

    # Create the email
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        # SSL port for Gmail
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False
        
# ---------- App UI ----------
st.set_page_config(page_title="Patient Health Monitoring", layout="wide")
st.title("🏥 Patient Health Monitoring")

# Load model
model = load_model()

# Ensure users DB exists
ensure_users_csv()

# Sidebar: choose action
with st.sidebar:
    st.header("Account")
    menu = st.selectbox("Go to", ["Login", "Sign up", "About"])

# ---------- SIGNUP ----------
if menu == "Sign up":
    st.subheader("Create a new account")
    new_name = st.text_input("Full name", key="signup_name")
    new_username = st.text_input("Username", key="signup_user")
    new_password = st.text_input("Password", type="password", key="signup_pass")
    new_caretaker = st.text_input("Caretaker / Doctor email", key="signup_caretaker")

    if st.button("Create account"):
        if not (new_name and new_username and new_password and new_caretaker):
            st.warning("Please fill all fields.")
        else:
            users = read_users()
            if new_username in users["username"].values:
                st.error("Username already exists — choose another.")
            else:
                hashed = hash_password(new_password)
                ok, msg = add_user(new_username, new_name, hashed, new_caretaker)
                if ok:
                    st.success("Account created. Now go to Login.")
                else:
                    st.error(msg)

# ---------- LOGIN ----------
elif menu == "Login":
    st.subheader("Login to your account")
    login_username = st.text_input("Username", key="login_user")
    login_password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Login"):
        users = read_users()
        row = users.loc[users["username"] == login_username]
        if row.empty:
            st.error("User not found. Please sign up.")
        else:
            stored = row["password"].values[0]
            if verify_password(stored, login_password):
                st.success(f"Welcome {row['name'].values[0]}!")
                # set session so main app knows user is logged in
                st.session_state["user"] = login_username
                st.session_state["name"] = row["name"].values[0]
                st.rerun()
            else:
                st.error("Incorrect password.")

# ---------- MAIN APP (after login) ----------
if "user" in st.session_state:
    username = st.session_state["user"]
    name = st.session_state.get("name", username)
    st.sidebar.write(f"Logged in as: {name}")
    if st.sidebar.button("Logout"):
        for k in ["user", "name"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    st.subheader("Enter patient health details")

    # Default values (you can adjust if needed)
    vals = [80, 16, 37.0, 97, 120, 80, 30, "Male", 22.0, 90]

    # Inputs (order must match training)
    heart_rate = st.number_input("Heart Rate (bpm)", 30, 200, int(vals[0]))
    resp_rate = st.number_input("Respiratory Rate (breaths/min)", 8, 60, int(vals[1]))
    body_temp = st.number_input("Body Temperature (°C)", 30.0, 45.0, float(vals[2]))
    oxygen_sat = st.number_input("Oxygen Saturation (%)", 50, 100, int(vals[3]))
    sys_bp = st.number_input("Systolic Blood Pressure (mmHg)", 60, 200, int(vals[4]))
    dia_bp = st.number_input("Diastolic Blood Pressure (mmHg)", 40, 140, int(vals[5]))
    age = st.number_input("Age", 0, 120, int(vals[6]))
    gender = st.selectbox("Gender", ["Male", "Female"], index=0 if vals[7] == "Male" else 1)
    derived_bmi = st.number_input("Derived BMI", 10.0, 50.0, float(vals[8]))
    derived_map = st.number_input("Derived MAP", 50.0, 140.0, float(vals[9]))

    # Encode gender same as training: (Male=0, Female=1)
    gender_encoded = 0 if gender == "Male" else 1

    features_array = np.array([[heart_rate, resp_rate, body_temp, oxygen_sat,
                                sys_bp, dia_bp, age, gender_encoded, derived_bmi, derived_map]])

    patient_data = {
        "Heart Rate": heart_rate,
        "Respiratory Rate": resp_rate,
        "Body Temperature": body_temp,
        "Oxygen Saturation": oxygen_sat,
        "Systolic Blood Pressure": sys_bp,
        "Diastolic Blood Pressure": dia_bp,
        "Age": age,
        "Gender": gender,
        "Derived_BMI": derived_bmi,
        "Derived_MAP": derived_map
    }
    
    

    if st.button("Predict Risk"):
        pred = model.predict(features_array)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(features_array).max()
            risk_text = "High Risk" if pred == 1 else "Low Risk"

        # --- Risk Breakdown Logic ---
        normal_ranges = {
            "Heart Rate": (60, 100),
            "Respiratory Rate": (12, 20),
            "Body Temperature": (36.1, 37.2),
            "Oxygen Saturation": (95, 100),
            "Systolic Blood Pressure": (90, 120),
            "Diastolic Blood Pressure": (60, 80),
            "Age": (0, 120),  # Age not used for abnormal flag here
            "Derived_BMI": (18.5, 24.9),
            "Derived_MAP": (70, 100)
        }

        abnormal_vitals = []
        for vital, (low, high) in normal_ranges.items():
            value = patient_data[vital]
            if isinstance(value, (int, float)) and not (low <= value <= high):
                abnormal_vitals.append(f"{vital}: {value} (normal {low}-{high})")

        # --- Display Result ---
        if pred == 1:
            st.error(f"⚠ {risk_text}  " + (f"(confidence={prob:.2f})" if prob is not None else ""))
            if abnormal_vitals:
                st.warning("Key factors for high risk:")
                for vital in abnormal_vitals:
                    st.write(f"🔴 {vital}")
            # Email alert
            users_df = read_users()
            caretaker_email = users_df.loc[users_df["username"] == username, "caretaker_email"].values[0]
            sent = send_email_alert(patient_data, caretaker_email)
            if sent:
                st.info(f"Email alert sent to {caretaker_email}")
        else:
            st.success(f"✅ {risk_text}  " + (f"(confidence={prob:.2f})" if prob is not None else ""))
            if not abnormal_vitals:
                st.info("All vitals are within safe ranges ✅")
            else:
                st.warning("Some vitals are outside normal range but not critical:")
                for vital in abnormal_vitals:
                    st.write(f"🟠 {vital}")

        # Save to history
        row = {
            "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **patient_data,
            "Risk": risk_text,
            "Abnormal Vitals": ", ".join(abnormal_vitals) if abnormal_vitals else "None"
            }
        save_history(username, row)
        st.info("Saved prediction to your history.")
        
    st.markdown("---")
    st.subheader("Your patient history")
    history_df = load_history(username)
    if history_df is not None and not history_df.empty:
        st.markdown("## 🩺 Patient Health Report")
        st.markdown(f"*Patient Name:* {name}  |  *Username:* {username}")
        st.markdown(f"*Report Generated On:* {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # On-screen summary stats (unchanged)
        summary_stats = history_df.describe().T[["mean", "min", "max"]]
        summary_stats = summary_stats.rename(columns={
            "mean": "Average", "min": "Minimum", "max": "Maximum"
        }).round(2)

        st.markdown("### 📈 Summary of Vitals")
        st.dataframe(summary_stats)

        # Professional vitals chart
        history_df["Date"] = pd.to_datetime(history_df["Date"], errors="coerce")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=history_df, x="Date", y="Heart Rate", marker="o", label="Heart Rate (bpm)", ax=ax)
        sns.lineplot(data=history_df, x="Date", y="Oxygen Saturation", marker="o", label="Oxygen Saturation (%)", ax=ax)
        sns.lineplot(data=history_df, x="Date", y="Body Temperature", marker="o", label="Body Temp (°C)", ax=ax)
        ax.set_title("Vital Signs Over Time", fontsize=14, fontweight="bold")
        ax.set_ylabel("Measurements")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Risk distribution
        fig2, ax2 = plt.subplots()
        sns.countplot(data=history_df, x="Risk", palette="coolwarm", ax=ax2)
        ax2.set_title("Risk Assessment Count", fontsize=14, fontweight="bold")
        st.pyplot(fig2)

        # Final assessment
        high_risk_count = (history_df["Risk"] == "High Risk").sum()
        low_risk_count = (history_df["Risk"] == "Low Risk").sum()

        st.markdown("### 🧾 Final Medical Summary")
        if high_risk_count > low_risk_count:
            st.error(f"⚠ Patient shows *frequent high-risk* readings ({high_risk_count} times). Immediate medical review recommended.")
        else:
            st.success(f"✅ Patient shows *mostly stable* readings ({low_risk_count} times low risk). Continue regular monitoring.")

        # CSV Download (unchanged)
        csv_bytes = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📄 Download Full Patient Report (CSV)",
            csv_bytes,
            file_name=f"patient_report_{username}.csv"
        )
        # Generate PDF without cutting rows
        # --- Line-by-line PDF download ---
        # Get only the latest entry for PDF
        latest_entry_df = history_df.tail(1)

        pdf_buffer = generate_pdf_report_line_by_line(latest_entry_df, name, username)
        st.download_button(
            "📄 Download Patient Report (PDF)",
            data=pdf_buffer,
            file_name=f"report_{username}.pdf",
            mime="application/pdf",
        )
        
    else:
        st.write("No history yet for this account. Predictions will be stored here.")
        
    # ---------- ABOUT ----------
elif menu == "About":
    st.write(""" Patient Health Monitoring App
    - Multi-user with secure hashed passwords (stored locally in users.csv)
    - Per-user history saved locally (history_<username>.csv)
    - Email alerts to caretaker email configured during signup (Gmail App Password recommended)
    - Uses trained ML model file: health_model.pkl
    """)