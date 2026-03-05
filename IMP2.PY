from flask import Flask, render_template_string, redirect, request, session, jsonify, Response
import threading, webbrowser, qrcode, io, base64, csv, sqlite3, os
from datetime import datetime
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
app.secret_key = "secure123"
DB = "attendance.db"
FACE_FOLDER = "faces"

os.makedirs(FACE_FOLDER, exist_ok=True)

# ==================== DATABASE ====================
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    # Ensure admin and student tables support username + password
    c.execute("""CREATE TABLE IF NOT EXISTS admin (
                    id INTEGER PRIMARY KEY,
                    username TEXT,
                    password TEXT NOT NULL
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS student (
                    id INTEGER PRIMARY KEY,
                    username TEXT,
                    password TEXT NOT NULL
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY,
                    unit_code TEXT NOT NULL,
                    week INTEGER NOT NULL,
                    reg_no TEXT NOT NULL,
                    student_name TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(unit_code, week, reg_no)
                )""")

    # Migrate existing tables if 'username' column is missing
    try:
        for table in ("admin", "student"):
            c.execute(f"PRAGMA table_info({table})")
            cols = [r[1] for r in c.fetchall()]
            if 'username' not in cols:
                c.execute(f"ALTER TABLE {table} ADD COLUMN username TEXT")
    except Exception:
        pass

    conn.commit()
    conn.close()

init_db()

# ==================== FACE RECOGNITION ENGINE ====================
class FaceRecognizer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.orb = cv2.ORB_create(nfeatures=500)

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(50, 50))
        return faces, gray

    def extract_face(self, image, face_rect):
        x, y, w, h = face_rect
        face = image[y:y+h, x:x+w]
        return cv2.resize(face, (200, 200))

    def compute_features(self, face_image):
        kp, des = self.orb.detectAndCompute(cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY), None)
        return kp, des

    def match_faces(self, ref_face, new_face):
        gray_ref = cv2.cvtColor(ref_face, cv2.COLOR_BGR2GRAY)
        gray_new = cv2.cvtColor(new_face, cv2.COLOR_BGR2GRAY)

        # SSIM comparison
        ssim_score = ssim(gray_ref, gray_new)

        # ORB feature matching
        kp1, des1 = self.orb.detectAndCompute(gray_ref, None)
        kp2, des2 = self.orb.detectAndCompute(gray_new, None)

        if des1 is None or des2 is None:
            return ssim_score * 0.8  # fallback to SSIM only

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 10:
            match_score = 0.8
        elif len(matches) > 5:
            match_score = 0.5
        else:
            match_score = 0

        # Weighted score
        return (ssim_score * 0.6) + (match_score * 0.4)

face_recognizer = FaceRecognizer()

# ==================== CSS STYLES ====================
STYLE = """
<style>
*{box-sizing:border-box;}
body{font-family:'Segoe UI',Arial; margin:0; padding:0; background:#f0f2f5;}
header{padding:15px; color:white; font-size:20px; text-align:center;}
.admin-header{background:#e74c3c;}
.student-header{background:#27ae60;}
.panel{margin:20px auto; width:90%; max-width:900px; background:white; padding:20px; border-radius:10px; box-shadow:0 0 10px rgba(0,0,0,0.1);}
h2{margin-top:0; color:#333;}
h3{color:#27ae60;}
button{padding:12px 24px; background:#3498db; color:white; border:none; border-radius:5px; cursor:pointer; font-weight:bold; transition:0.3s;}
button:hover{background:#2980b9;}
.card{background:#ecf0f1; padding:15px; border-radius:8px; margin-bottom:10px;}
input{padding:10px; width:100%; margin:5px 0; border-radius:5px; border:1px solid #bdc3c7;}
input:focus{outline:none;border:2px solid #3498db;}
table{width:100%; border-collapse:collapse; margin-top:10px;}
th, td{border:1px solid #ccc; padding:8px; text-align:left;}
th{background:#3498db; color:white;}
.qr{width:200px; height:200px; margin-top:10px; border:2px solid #333; border-radius:10px;}
video{border:2px solid #27ae60; border-radius:10px; width:100%; max-width:350px; background:#000;}
.status{margin:15px 0; padding:10px; border-radius:5px; font-weight:bold;}
.success{background:#d4edda; color:#155724; border:1px solid #c3e6cb;}
.error{background:#f8d7da; color:#721c24; border:1px solid #f5c6cb;}
.info{background:#d1ecf1; color:#0c5460; border:1px solid #bee5eb;}
#scanner-container{text-align:center;}
.scan-hint{font-size:14px; color:#666; margin:10px 0; padding:10px; background:#f9f9f9; border-radius:5px;}
.progress-bar{width:100%; height:20px; background:#ecf0f1; border-radius:5px; overflow:hidden; margin:10px 0;}
.progress-fill{height:100%; background:#3498db; width:0%; transition:width 0.3s;}
</style>
"""

# ==================== HOME ====================
@app.route("/")
def home():
    return render_template_string(STYLE+"""
<div class="panel">
<h2>🎯 Smart Attendance System</h2>
<p><button onclick="location.href='/admin-login'" style="width:100%;padding:15px;">👨‍🏫 Admin Login</button></p>
<p><button onclick="location.href='/student-login'" style="width:100%;padding:15px;">🎓 Student Login</button></p>
</div>
""")

# ==================== PASSWORD HELPERS ====================
def get_credentials(table):
    try:
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute(f"SELECT username, password FROM {table} LIMIT 1")
        row = c.fetchone()
        conn.close()
        if row:
            return (row[0], row[1])
        return (None, None)
    except Exception:
        return (None, None)

def set_credentials(table, username, password):
    try:
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute(f"DELETE FROM {table}")
        c.execute(f"INSERT INTO {table} (username, password) VALUES (?,?)", (username, password))
        conn.commit()
        conn.close()
    except Exception as e:
        print("Error:", e)

# ==================== ADMIN & STUDENT LOGIN ====================
@app.route("/admin-login", methods=["GET","POST"])
def admin_login():
    error=""
    saved_user, saved_pw = get_credentials("admin")
    first_time = (saved_user is None and saved_pw is None)
    if request.method=="POST":
        username = request.form.get("username","").strip()
        pw = request.form.get("password","").strip()
        if first_time:
            other_user, _ = get_credentials("student")
            if other_user and username == other_user:
                error = "❌ This username is already used by a student. Choose another."
            else:
                set_credentials("admin", username, pw)
                session["admin"] = True
                session["admin_username"] = username
                return redirect("/admin")
        else:
            if username == (saved_user or "") and pw == (saved_pw or ""):
                session["admin"] = True
                session["admin_username"] = username
                return redirect("/admin")
            else:
                error="❌ Invalid username or password"
    return render_template_string(STYLE+f"""
<div class="panel">
<h2>Admin Login</h2>
<form method="POST">
<input name="username" type="text" required><br>
<input name="password" type="password" required><br>
<button>Submit</button>
</form>
<p style="color:red;">{error}</p>
<p style="margin-top:8px;"><a href="/forgot-password/admin">Forgot password?</a></p>
<p style="margin-top:8px;"><a href="/"><button>← Back</button></a></p>
</div>
""")

@app.route("/student-login", methods=["GET","POST"])
def student_login():
    error=""
    saved_user, saved_pw = get_credentials("student")
    first_time = (saved_user is None and saved_pw is None)
    if request.method=="POST":
        username = request.form.get("username","").strip()
        pw = request.form.get("password","").strip()
        if first_time:
            other_user, _ = get_credentials("admin")
            if other_user and username == other_user:
                error = "❌ This username is already used by an admin. Choose another."
            else:
                set_credentials("student", username, pw)
                session["student"] = True
                session["student_username"] = username
                return redirect("/student")
        else:
            if username == (saved_user or "") and pw == (saved_pw or ""):
                session["student"] = True
                session["student_username"] = username
                return redirect("/student")
            else:
                error="❌ Invalid username or password"
    return render_template_string(STYLE+f"""
<div class="panel">
<h2>Student Login</h2>
<form method="POST">
<input name="username" type="text" required><br>
<input name="password" type="password" required><br>
<button>Submit</button>
</form>
<p style="color:red;">{error}</p>
<p style="margin-top:8px;"><a href="/forgot-password/student">Forgot password?</a></p>
<p style="margin-top:8px;"><a href="/"><button>← Back</button></a></p>
</div>
""")

@app.route('/forgot-password/<role>', methods=['GET','POST'])
def forgot_password(role):
    if role not in ('admin', 'student'):
        return redirect('/')
    msg = ''
    saved_user, saved_pw = get_credentials(role)
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        pw = request.form.get('password','').strip()
        confirm = request.form.get('confirm','').strip()
        if not username or not pw:
            msg = '❌ Username and password are required.'
        elif pw != confirm:
            msg = '❌ Passwords do not match.'
        else:
            if saved_user and saved_user != username:
                msg = '❌ Username does not match our records.'
            else:
                other_role = 'student' if role == 'admin' else 'admin'
                other_user, _ = get_credentials(other_role)
                if other_user and other_user == username and other_role != role:
                    msg = f'❌ This username is already used by {other_role}. Choose another.'
                else:
                    set_credentials(role, username, pw)
                    return redirect(f'/{role}-login')

    role_title = 'Admin' if role == 'admin' else 'Student'
    return render_template_string(STYLE + f"""
<div class="panel">
<h2>{role_title} Password Reset</h2>
<form method="POST">
<input name="username" type="text" placeholder="Username" required><br>
<input name="password" type="password" placeholder="New password" required><br>
<input name="confirm" type="password" placeholder="Confirm new password" required><br>
<button>Reset Password</button>
</form>
<p style="color:red;">{msg}</p>
<p style="margin-top:8px;"><a href='/{role}-login'>← Back to Login</a></p>
</div>
""")

# ==================== ADMIN DASHBOARD ====================
ADMIN_PAGE = STYLE + """
<header class="admin-header">👨‍🏫 ADMIN DASHBOARD</header>
<div class="panel">

<div class="card">
<h3>📱 Generate QR for Unit</h3>
<form method="POST" action="/generate-qr">
<input name="unit_code" placeholder="Enter Unit Code" required><br>
<input name="week" type="number" placeholder="Enter Week Number" required><br>
<label>Class Duration (minutes):</label>
<input name="class_duration" type="number" value="120" min="30" max="300" required><br>
<button type="submit">Generate QR</button>
</form>
{% if qr_image %}
<img src="{{ qr_image }}" class="qr"><br>
<p style="background:#fff3cd;padding:10px;border-radius:5px;margin:10px 0;">
  ⏰ <strong>QR Validity:</strong> Valid for {{ class_duration }} minutes from generation<br>
  🔒 <strong>Security:</strong> QR expires after class duration + Face verification required
</p>
<a href="{{ qr_image }}" download="qr_{{unit_code}}_week{{week}}.png"><button>Download QR</button></a>
{% endif %}
</div>

<div class="card">
<h3>📊 Attendance Log</h3>
<div id="attendance-log">Loading...</div>
</div>

<button onclick="location.href='/logout'" style="width:100%;padding:15px;background:#e74c3c;">Logout</button>
</div>

<script>
async function loadData(){
    try{
        const resp = await fetch('/live-data');
        const d = await resp.json();
        let logDiv = "<h3>📊 Attendance Log</h3>";
        for(let unit_week in d.attendance){
            let count = d.attendance[unit_week].length;
            logDiv += "<strong>"+unit_week+"</strong>: "+count+" students <a href='/download-attendance/"+encodeURIComponent(unit_week)+"'><button style='padding:8px 16px;font-size:14px;'>Download CSV</button></a><br>";
        }
        document.getElementById("attendance-log").innerHTML = logDiv;
    }catch(e){console.error(e);}
}
setInterval(loadData, 2000);
loadData();
</script>
"""

@app.route("/admin")
def admin():
    if not session.get("admin"):
        return redirect("/")
    return render_template_string(ADMIN_PAGE, qr_image=None, attendance={})

# ==================== GENERATE QR ====================
@app.route("/generate-qr", methods=["POST"])
def generate_qr():
    if not session.get("admin"):
        return redirect("/")
    unit_code = request.form.get("unit_code","").strip()
    week = request.form.get("week","").strip()
    class_duration = request.form.get("class_duration", "120").strip()  # minutes

    qr_timestamp = datetime.now().timestamp()
    qr_text = f"{unit_code}|{week}|{int(qr_timestamp)}|{class_duration}"

    qr = qrcode.QRCode(box_size=8, border=2)
    qr.add_data(qr_text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    qr_data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    if not hasattr(app, 'valid_qr_codes'):
        app.valid_qr_codes = {}
    app.valid_qr_codes[f"{unit_code}_{week}"] = {
        "timestamp": qr_timestamp,
        "duration": int(class_duration)
    }

    return render_template_string(ADMIN_PAGE, qr_image=qr_data_url, unit_code=unit_code, week=week, attendance={})

# ==================== STUDENT DASHBOARD ====================
STUDENT_PAGE = STYLE + """
<header class="student-header">🎓 STUDENT DASHBOARD</header>
<div class="panel">

<div class="card">
<h3>🔄 Select Attendance Method</h3>
<button id="qr-btn" onclick="goQRScan()" style="width:100%;padding:15px;margin:5px 0;background:#9b59b6;">
  📱 Step 1: Scan QR Code
</button>
<button id="face-btn" onclick="goFaceScan()" style="width:100%;padding:15px;margin:5px 0;background:#1abc9c;">
  👤 Step 2: Face Verification
</button>
</div>

<button onclick="location.href='/logout'" style="width:100%;padding:15px;background:#e74c3c;">Logout</button>
</div>

<script>
function goQRScan(){ location.href='/qr-scanner'; }
function goFaceScan(){ location.href='/face-scanner'; }
</script>
"""

@app.route("/student")
def student():
    if not session.get("student"):
        return redirect("/")
    return render_template_string(STUDENT_PAGE)

# ... (QR scanner, face scanner, detection, attendance submission, download, live_data,
# logout routes remain unchanged except for removal of bluetooth checks)

# [Truncated here for brevity – the remainder of your original code continues verbatim
# from the QR_SCANNER_PAGE definition onward, omitting any bluetooth references as in
# the previous segments above.]

# ==================== RUN ====================
def open_browser():
    webbrowser.open("http://localhost:5000")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    is_local = os.environ.get('PORT') is None
    if is_local:
        threading.Timer(2, open_browser).start()
    app.run(debug=False, threaded=True, use_reloader=False, host="0.0.0.0", port=port)
