from flask import Flask, render_template_string, redirect, request, session, jsonify, Response
import threading, webbrowser, qrcode, io, base64, csv, sqlite3, os
from datetime import datetime
from bleak import BleakScanner
import asyncio
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

# ==================== BLUETOOTH SCANNING ====================
bluetooth_devices = 0
bluetoothEnabled = False

async def scan_bt():
    global bluetooth_devices, bluetoothEnabled
    try:
        devices = await BleakScanner.discover(timeout=5.0)
        bluetooth_devices = len(devices)
        bluetoothEnabled = True  # If scan succeeds, Bluetooth is ON
    except Exception as e:
        print(f"Bluetooth error: {e}")
        bluetooth_devices = 0
        bluetoothEnabled = False  # Bluetooth is OFF or unavailable

def bt_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        try:
            loop.run_until_complete(scan_bt())
        except Exception as e:
            print(f"BT Loop error: {e}")
        loop.run_until_complete(asyncio.sleep(3))

threading.Thread(target=bt_loop, daemon=True).start()

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

# ==================== ADMIN LOGIN ====================
@app.route("/admin-login", methods=["GET","POST"])
def admin_login():
    error=""
    saved_user, saved_pw = get_credentials("admin")
    first_time = (saved_user is None and saved_pw is None)
    if request.method=="POST":
        username = request.form.get("username","").strip()
        pw = request.form.get("password","").strip()
        if first_time:
            # Prevent admin username matching existing student username
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

# ==================== STUDENT LOGIN ====================
@app.route("/student-login", methods=["GET","POST"])
def student_login():
    error=""
    saved_user, saved_pw = get_credentials("student")
    first_time = (saved_user is None and saved_pw is None)
    if request.method=="POST":
        username = request.form.get("username","").strip()
        pw = request.form.get("password","").strip()
        if first_time:
            # Prevent student username matching existing admin username
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
    # minimal, user-driven password reset for admin or student
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
            # If a username exists in DB, require it to match the provided username
            if saved_user and saved_user != username:
                msg = '❌ Username does not match our records.'
            else:
                # Prevent resetting/creating a username that is already used by the other role
                other_role = 'student' if role == 'admin' else 'admin'
                other_user, _ = get_credentials(other_role)
                if other_user and other_user == username and other_role != role:
                    msg = f'❌ This username is already used by {other_role}. Choose another.'
                else:
                    set_credentials(role, username, pw)
                    # After reset/creation, redirect to the appropriate login
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
<h3>📱 Generate QR for Unit (WITH CLASS VERIFICATION)</h3>
<form method="POST" action="/generate-qr">
<input name="unit_code" placeholder="Enter Unit Code" required><br>
<input name="week" type="number" placeholder="Enter Week Number" required><br>
<label>Class Duration (minutes):</label>
<input name="class_duration" type="number" value="120" min="30" max="300" required><br>
<label>Minimum Bluetooth Devices Required (for class attendance verification):</label>
<input name="bluetooth_required" type="number" value="3" min="1" max="10" required>
<p style="font-size:13px;color:#666;">💡 Set this to classroom Bluetooth devices count (teacher device + classroom equipment)</p>
<button type="submit">Generate QR</button>
</form>
{% if qr_image %}
<img src="{{ qr_image }}" class="qr"><br>
<p style="background:#fff3cd;padding:10px;border-radius:5px;margin:10px 0;">
  ⏰ <strong>QR Validity:</strong> Valid for {{ class_duration }} minutes from generation<br>
  📡 <strong>Bluetooth Check:</strong> Requires {{ bluetooth_required }}+ devices in classroom<br>
  🔒 <strong>Security:</strong> QR expires after class duration + Face verification required
</p>
<a href="{{ qr_image }}" download="qr_{{unit_code}}_week{{week}}.png"><button>Download QR</button></a>
{% endif %}
</div>

<div class="card">
<h3>📡 Bluetooth Devices Nearby</h3>
<div id="bluetooth">Loading...</div>
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
        document.getElementById("bluetooth").innerText = d.bluetooth + " devices nearby";
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
    
    # Generate QR with timestamp so it expires after class
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
    
    # Store valid QR codes in memory for classroom verification
    if not hasattr(app, 'valid_qr_codes'):
        app.valid_qr_codes = {}
    app.valid_qr_codes[f"{unit_code}_{week}"] = {
        "timestamp": qr_timestamp,
        "duration": int(class_duration),
        "bluetooth_required": int(request.form.get("bluetooth_required", "3"))
    }
    
    return render_template_string(ADMIN_PAGE, qr_image=qr_data_url, unit_code=unit_code, week=week, attendance={})

# ==================== STUDENT DASHBOARD (SEPARATE FLOWS) ====================
STUDENT_PAGE = STYLE + """
<header class="student-header">🎓 STUDENT DASHBOARD</header>
<div class="panel">

<div class="card">
<h3>📡 Bluetooth Status</h3>
<div id="bluetooth-status" style="font-size:16px;font-weight:bold;">Checking...</div>
</div>

<div class="card">
<h3>🔄 Select Attendance Method</h3>
<p id="bt-warning" style="color:#e74c3c; font-weight:bold; display:none;">⚠️ BLUETOOTH MUST BE ON to mark attendance</p>
<button id="qr-btn" onclick="goQRScan()" style="width:100%;padding:15px;margin:5px 0;background:#9b59b6;" disabled>
  📱 Step 1: Scan QR Code
</button>
<button id="face-btn" onclick="goFaceScan()" style="width:100%;padding:15px;margin:5px 0;background:#1abc9c;" disabled>
  👤 Step 2: Face Verification
</button>
</div>

<button onclick="location.href='/logout'" style="width:100%;padding:15px;background:#e74c3c;">Logout</button>
</div>

<script>
function goQRScan(){
    location.href='/qr-scanner';
}
function goFaceScan(){
    location.href='/face-scanner';
}

async function updateBluetooth() {
    try{
        const resp = await fetch('/bluetooth-status');
        const data = await resp.json();
        const statusDiv = document.getElementById("bluetooth-status");
        const warningDiv = document.getElementById("bt-warning");
        const qrBtn = document.getElementById("qr-btn");
        const faceBtn = document.getElementById("face-btn");
        
        if(data.enabled){
            statusDiv.innerHTML = "✅ <span style='color:#27ae60;'>ENABLED</span> - " + data.devices + " devices nearby";
            statusDiv.style.color = "#27ae60";
            warningDiv.style.display = "none";
            qrBtn.disabled = false;
            faceBtn.disabled = false;
            qrBtn.style.opacity = "1";
            faceBtn.style.opacity = "1";
        }else{
            statusDiv.innerHTML = "❌ <span style='color:#e74c3c;'>DISABLED</span>";
            statusDiv.style.color = "#e74c3c";
            warningDiv.style.display = "block";
            qrBtn.disabled = true;
            faceBtn.disabled = true;
            qrBtn.style.opacity = "0.5";
            faceBtn.style.opacity = "0.5";
        }
    }catch(e){console.error(e);}
}
setInterval(updateBluetooth, 2000);
updateBluetooth();
</script>
"""

@app.route("/student")
def student():
    if not session.get("student"):
        return redirect("/")
    return render_template_string(STUDENT_PAGE)

# ==================== QR SCANNER (STEP 1) - OPTIMIZED ====================
QR_SCANNER_PAGE = STYLE + """
<header class="student-header">📱 QR CODE SCANNER</header>
<div class="panel">
<div class="card">
<h3>Point camera at QR Code</h3>
<div class="scan-hint">
  💡 <strong>Tips for faster scanning:</strong><br>
  • Works with QR codes on paper OR device screens<br>
  • Ensure QR code is well-lit and clear<br>
  • Hold device steady and centered<br>
  • QR code should be 15-30cm from camera<br>
  • If scanning from screen, reduce screen brightness if glare occurs
</div>
<div id="scanner-container">
<video id="qr-video" playsinline></video>
</div>
<div class="progress-bar">
  <div class="progress-fill" id="scan-progress"></div>
</div>
<p id="qr-status" class="status info">⏳ Initializing camera...</p>
<p id="qr-debug" style="font-size:12px;color:#666;"></p>
<button onclick="backToMenu()">← Back to Menu</button>
</div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/jsqr/1.4.0/jsQR.min.js"></script>
<script>
const qrVideo = document.getElementById("qr-video");
const qrStatus = document.getElementById("qr-status");
const qrDebug = document.getElementById("qr-debug");
const scanProgress = document.getElementById("scan-progress");
let qrDetected = null;
let frameCount = 0;
let startTime = Date.now();
const SCAN_TIMEOUT = 30000; // 30 second timeout

async function initQRScanner(){
    try{
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                facingMode: "environment",
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        });
        qrVideo.srcObject = stream;
        qrVideo.setAttribute("playsinline", true);
        qrVideo.style.display = "block";
        qrVideo.play();
        qrStatus.textContent = "✅ Camera ready - scanning for QR code...";
        qrStatus.className = "status info";
        qrDebug.textContent = "Scanning...";
        scanQRLoop();
    }catch(err){
        qrStatus.textContent = "❌ Cannot access camera: " + err.message;
        qrStatus.className = "status error";
        qrDebug.textContent = "Camera access denied";
    }
}

function scanQRLoop(){
    if(qrDetected) return;
    
    // Check timeout
    const elapsed = Date.now() - startTime;
    if(elapsed > SCAN_TIMEOUT){
        qrStatus.innerHTML = "<strong>❌ QR SCAN TIMEOUT</strong><br>Unable to detect QR code after 30 seconds.<br>Possible issues:<br>• QR code is not clear or focused<br>• Lighting is too dark<br>• QR code is damaged or unreadable<br><br><a onclick='location.reload()' style='color:#721c24;cursor:pointer;text-decoration:underline;'>Click to retry</a>";
        qrStatus.className = "status error";
        qrDebug.textContent = "Timeout - Please check QR code quality and lighting";
        stopQRCamera();
        return;
    }
    
    // Update progress bar
    let progress = (elapsed / SCAN_TIMEOUT) * 100;
    scanProgress.style.width = progress + "%";
    
    if(qrVideo.readyState === qrVideo.HAVE_ENOUGH_DATA){
        const canvas = document.createElement("canvas");
        canvas.width = qrVideo.videoWidth;
        canvas.height = qrVideo.videoHeight;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(qrVideo, 0, 0, canvas.width, canvas.height);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const code = jsQR(imageData.data, imageData.width, imageData.height, { inversionAttempts: "attemptBoth" });
        
        frameCount++;
        
        if(code){
            qrDetected = code.data;
            qrStatus.textContent = "✅ QR CODE DETECTED! Processing...";
            qrStatus.className = "status success";
            qrDebug.textContent = "QR Code Data: " + qrDetected;
            scanProgress.style.width = "100%";
            stopQRCamera();
            sessionStorage.setItem("qr_code", qrDetected);
            setTimeout(() => location.href='/face-scanner', 1200);
            return;
        }
        
        // Update status every 5 frames to reduce overhead
        if(frameCount % 5 === 0){
            qrStatus.textContent = "👁️ Scanning... (" + Math.round(elapsed/1000) + "s)";
            qrDebug.textContent = "Frames processed: " + frameCount + " | Elapsed: " + Math.round(elapsed/1000) + "s";
        }
    }
    requestAnimationFrame(scanQRLoop);
}

function stopQRCamera(){
    if(qrVideo.srcObject){
        qrVideo.srcObject.getTracks().forEach(track => track.stop());
    }
    qrVideo.style.display = "none";
}

function backToMenu(){
    stopQRCamera();
    sessionStorage.removeItem("qr_code");
    location.href='/student';
}

initQRScanner();
</script>
"""

@app.route("/qr-scanner")
def qr_scanner():
    if not session.get("student"):
        return redirect("/")
    if not bluetoothEnabled:
        return render_template_string(STYLE + """
<div class="panel">
<div class="status error">
<h2>❌ BLUETOOTH DISABLED</h2>
<p>Your device's Bluetooth is currently OFF.</p>
<p>Please enable Bluetooth to proceed with attendance marking.</p>
<p><a href='/student'><button>← Back to Dashboard</button></a></p>
</div>
</div>
""")
    return render_template_string(QR_SCANNER_PAGE)

# ==================== FACE SCANNER (STEP 2) ====================
FACE_SCANNER_PAGE = STYLE + """
<header class="student-header">👤 FACE VERIFICATION</header>
<div class="panel">
<div class="card">
<h3>Position your face in frame</h3>
<div class="scan-hint">
  💡 <strong>For best results:</strong><br>
  • Face the camera directly<br>
  • Good lighting is important<br>
  • Keep your entire face visible<br>
  • Face will capture automatically when detected
</div>
<div id="scanner-container">
<video id="face-video" playsinline></video>
</div>
<p id="face-status" class="status info">⏳ Initializing camera...</p>
<p id="face-debug" style="font-size:12px;color:#666;"></p>
<button onclick="backToMenu()">← Back to Menu</button>
</div>
</div>

<script>
const faceVideo = document.getElementById("face-video");
const faceStatus = document.getElementById("face-status");
const faceDebug = document.getElementById("face-debug");
let faceCaptured = null;
let captureAttempts = 0;

async function initFaceScanner(){
    try{
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });
        faceVideo.srcObject = stream;
        faceVideo.setAttribute("playsinline", true);
        faceVideo.style.display = "block";
        faceVideo.play();
        faceStatus.textContent = "✅ Camera ready - align your face...";
        faceStatus.className = "status info";
        detectFaceLoop();
    }catch(err){
        faceStatus.textContent = "❌ Cannot access camera: " + err.message;
        faceStatus.className = "status error";
    }
}

function detectFaceLoop(){
    if(faceVideo.readyState === faceVideo.HAVE_ENOUGH_DATA && !faceCaptured){
        const canvas = document.createElement("canvas");
        canvas.width = faceVideo.videoWidth;
        canvas.height = faceVideo.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(faceVideo, 0, 0, canvas.width, canvas.height);
        
        // Send to backend for face detection - check every 3 frames
        captureAttempts++;
        if(captureAttempts % 3 === 0){
            canvas.toBlob(blob => {
                checkFaceDetection(blob);
            }, "image/jpeg", 0.8);
        }
    }
    if(!faceCaptured){
        requestAnimationFrame(detectFaceLoop);
    }
}

async function checkFaceDetection(blob){
    const formData = new FormData();
    formData.append("frame", blob);
    
    try{
        const resp = await fetch("/detect-face", { method: "POST", body: formData });
        const result = await resp.json();
        
        if(result.detected){
            faceCaptured = true;
            faceStatus.textContent = "✅ FACE DETECTED! Capturing...";
            faceStatus.className = "status success";
            faceDebug.textContent = "Face captured successfully!";
            captureFace();
        }else{
            faceStatus.textContent = "👤 Detecting face... Please wait (" + result.attempts + " attempts)";
            faceStatus.className = "status info";
            if(result.message) faceDebug.textContent = result.message;
        }
    }catch(e){
        console.error(e);
        faceDebug.textContent = "Error checking face detection";
    }
}

function captureFace(){
    const canvas = document.createElement("canvas");
    canvas.width = faceVideo.videoWidth;
    canvas.height = faceVideo.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(faceVideo, 0, 0, canvas.width, canvas.height);
    const faceImage = canvas.toDataURL("image/jpeg", 0.9);
    
    stopFaceCamera();
    promptUserDetails(faceImage);
}

function stopFaceCamera(){
    if(faceVideo.srcObject){
        faceVideo.srcObject.getTracks().forEach(track => track.stop());
    }
    faceVideo.style.display = "none";
}

function promptUserDetails(faceImage){
    const regNo = prompt("Enter your Registration Number:");
    if(!regNo) { alert("Registration number is required!"); location.href='/student'; return; }
    
    const studentName = prompt("Enter your Full Name:");
    if(!studentName) { alert("Name is required!"); location.href='/student'; return; }
    
    submitAttendance(faceImage, regNo, studentName);
}

async function submitAttendance(faceImage, regNo, studentName){
    const qrCode = sessionStorage.getItem("qr_code");
    
    if(!qrCode){
        alert("❌ QR Code not found. Please scan QR first.");
        location.href='/qr-scanner';
        return;
    }
    
    faceStatus.textContent = "📤 Uploading and processing...";
    faceStatus.className = "status info";
    
    const formData = new FormData();
    formData.append("qr_code", qrCode);
    formData.append("reg_no", regNo);
    formData.append("student_name", studentName);
    formData.append("face_image", faceImage);
    
    try{
        const resp = await fetch("/submit-attendance", { method: "POST", body: formData });
        const text = await resp.text();
        document.open();
        document.write(text);
        document.close();
    }catch(e){
        faceStatus.textContent = "❌ Error: " + e.message;
        faceStatus.className = "status error";
        console.error(e);
    }
}

function backToMenu(){
    stopFaceCamera();
    sessionStorage.removeItem("qr_code");
    location.href='/student';
}

initFaceScanner();
</script>
"""

@app.route("/face-scanner")
def face_scanner():
    if not session.get("student"):
        return redirect("/")
    if not bluetoothEnabled:
        return render_template_string(STYLE + """
<div class="panel">
<div class="status error">
<h2>❌ BLUETOOTH DISABLED</h2>
<p>Your device's Bluetooth is currently OFF.</p>
<p>Please enable Bluetooth to proceed with attendance marking.</p>
<p><a href='/student'><button>← Back to Dashboard</button></a></p>
</div>
</div>
""")
    return render_template_string(FACE_SCANNER_PAGE)

# ==================== DETECT FACE (BACKEND) ====================
face_detection_attempts = {}

@app.route("/detect-face", methods=["POST"])
def detect_face():
    try:
        client_id = request.remote_addr
        if client_id not in face_detection_attempts:
            face_detection_attempts[client_id] = 0
        face_detection_attempts[client_id] += 1
        attempts = face_detection_attempts[client_id]
        
        # Save frame temporarily
        frame_data = request.files.get("frame")
        if not frame_data:
            return jsonify({"detected": False, "attempts": attempts, "message": "No frame received"})
        
        frame_bytes = np.frombuffer(frame_data.read(), np.uint8)
        image = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"detected": False, "attempts": attempts, "message": "Frame decode failed"})
        
        faces, _ = face_recognizer.detect_faces(image)
        detected = len(faces) > 0
        
        message = ""
        if detected:
            message = f"✅ Found {len(faces)} face(s)"
            face_detection_attempts[client_id] = 0
        else:
            message = "❌ No face detected. Check lighting and position."
        
        return jsonify({"detected": detected, "attempts": attempts, "message": message})
    except Exception as e:
        print(f"Face detection error: {e}")
        return jsonify({"detected": False, "attempts": 0, "message": "Error: " + str(e)})

# ==================== SUBMIT ATTENDANCE ====================
@app.route("/submit-attendance", methods=["POST"])
def submit_attendance():
    if not session.get("student"):
        return redirect("/")
    
    if not bluetoothEnabled:
        return f"""
{STYLE}
<div class="panel">
<div class="status error">
<h2>❌ BLUETOOTH DISABLED</h2>
<p>Bluetooth has been turned off. Attendance cannot be marked.</p>
<p>Please enable Bluetooth and try again.</p>
<p><a href='/student'><button>← Back to Dashboard</button></a></p>
</div>
</div>
"""
    
    try:
        qr = request.form.get("qr_code", "").strip()
        if "|" not in qr:
            return f"""
{STYLE}
<div class="panel">
<div class="status error">
<h2>❌ INVALID QR CODE</h2>
<p>The QR code format is invalid. Please scan a valid attendance QR code.</p>
<p><a href='/qr-scanner'><button>← Scan QR Again</button></a></p>
</div>
</div>
"""
        
        # Parse QR code with timestamp and duration
        qr_parts = qr.split("|")
        if len(qr_parts) < 2:
            return f"""
{STYLE}
<div class="panel">
<div class="status error">
<h2>❌ CORRUPTED QR CODE</h2>
<p>The QR code data is incomplete or corrupted.</p>
<p><a href='/qr-scanner'><button>← Scan QR Again</button></a></p>
</div>
</div>
"""
        
        unit_code = qr_parts[0]
        week = qr_parts[1]
        qr_timestamp = float(qr_parts[2]) if len(qr_parts) > 2 else None
        class_duration = int(qr_parts[3]) if len(qr_parts) > 3 else 120
        
        # SECURITY CHECK 1: QR CODE EXPIRATION - Must be within class duration
        if qr_timestamp:
            current_time = datetime.now().timestamp()
            elapsed_minutes = (current_time - qr_timestamp) / 60
            
            if elapsed_minutes > class_duration:
                return f"""
{STYLE}
<div class="panel">
<div class="status error">
<h2>❌ QR CODE EXPIRED</h2>
<p>This QR code is no longer valid.</p>
<p><strong>Class ended:</strong> {elapsed_minutes:.0f} minutes ago</p>
<p><strong>Allowed duration:</strong> {class_duration} minutes</p>
<p>⏰ <strong>You are LATE!</strong> Attendance marking window has closed.</p>
<p>Please contact your instructor if you have valid reasons.</p>
<p><a href='/student'><button>← Back to Dashboard</button></a></p>
</div>
</div>
"""
        
        # SECURITY CHECK 2: BLUETOOTH DEVICE REQUIREMENT - Must have classroom devices present
        min_devices_required = 3
        if not hasattr(app, 'valid_qr_codes'):
            app.valid_qr_codes = {}
        
        qr_key = f"{unit_code}_{week}"
        if qr_key in app.valid_qr_codes:
            min_devices_required = app.valid_qr_codes[qr_key].get("bluetooth_required", 3)
        
        if bluetooth_devices < min_devices_required:
            return f"""
{STYLE}
<div class="panel">
<div class="status error">
<h2>❌ NOT IN CLASSROOM - INSUFFICIENT BLUETOOTH DEVICES</h2>
<p>Your device is not detecting the required classroom Bluetooth devices.</p>
<p><strong>Detected devices:</strong> {bluetooth_devices}<br>
<strong>Required devices:</strong> {min_devices_required}</p>
<p>🚫 <strong>This indicates you are NOT in the classroom.</strong></p>
<p>Attendance cannot be marked remotely. Please:</p>
<ul style="text-align:left;display:inline-block;">
<li>Enter the classroom</li>
<li>Ensure classroom Bluetooth devices are active</li>
<li>Enable your device's Bluetooth</li>
<li>Try scanning again</li>
</ul>
<p><a href='/student'><button>← Back to Dashboard</button></a></p>
</div>
</div>
"""
        reg_no = request.form.get("reg_no", "").strip()
        student_name = request.form.get("student_name", "").strip()
        face_image = request.form.get("face_image", "")
        
        if not face_image or not face_image.startswith("data:"):
            return f"""
{STYLE}
<div class="panel">
<div class="status error">
<h2>❌ FACE IMAGE MISSING</h2>
<p>Face image was not captured properly.</p>
<p><a href='/face-scanner'><button>← Scan Face Again</button></a></p>
</div>
</div>
"""
        
        # Decode face image
        header, encoded = face_image.split(",", 1)
        face_data = base64.b64decode(encoded)
        face_file = f"temp_{reg_no}_{int(__import__('time').time())}.jpg"
        
        with open(face_file, "wb") as f:
            f.write(face_data)
        
        face_array = cv2.imread(face_file)
        if face_array is None:
            os.remove(face_file)
            return f"""
{STYLE}
<div class="panel">
<div class="status error">
<h2>❌ FACE IMAGE CORRUPTED</h2>
<p>The captured face image is corrupted or unreadable.</p>
<p><a href='/face-scanner'><button>← Scan Face Again</button></a></p>
</div>
</div>
"""
        
        # Detect faces in uploaded image
        faces, _ = face_recognizer.detect_faces(face_array)
        if len(faces) == 0:
            os.remove(face_file)
            return f"""
{STYLE}
<div class="panel">
<div class="status error">
<h2>❌ NO FACE DETECTED IN IMAGE</h2>
<p>The system could not detect your face in the captured image.</p>
<p><strong>Possible issues:</strong><br>
• Poor lighting conditions<br>
• Face is too small or at an angle<br>
• Part of face is obscured</p>
<p><a href='/face-scanner'><button>← Try Again</button></a></p>
</div>
</div>
"""
        
        if len(faces) > 1:
            os.remove(face_file)
            return f"""
{STYLE}
<div class="panel">
<div class="status error">
<h2>❌ MULTIPLE FACES DETECTED - FRAUD PREVENTION</h2>
<p>Only one person per scan is allowed.</p>
<p>🚨 <strong>Security Alert:</strong> Multiple faces detected in one scan.</p>
<p>Please ensure only your face is visible in the frame.</p>
<p><a href='/face-scanner'><button>← Try Again</button></a></p>
</div>
</div>
"""
        
        # Extract and crop face
        x, y, w, h = faces[0]
        face_crop = face_array[max(0, y-20):y+h+20, max(0, x-20):x+w+20]
        face_crop = cv2.resize(face_crop, (200, 200))
        
        ref_path = os.path.join(FACE_FOLDER, f"{reg_no}.jpg")
        
        # SECURITY CHECK 4: FIRST TIME REGISTRATION - Store face biometric
        if not os.path.exists(ref_path):
            cv2.imwrite(ref_path, face_crop)
            conn = sqlite3.connect(DB)
            c = conn.cursor()
            try:
                # Also store the registration details
                c.execute(
                    "INSERT INTO attendance (unit_code, week, reg_no, student_name) VALUES (?,?,?,?)",
                    (unit_code, int(week), reg_no, student_name)
                )
                conn.commit()
                os.remove(face_file)
                return f"""
{STYLE}
<div class="panel">
<div class="status success">
<h2>✅ REGISTERED & ATTENDANCE MARKED!</h2>
<p>Welcome <strong>{student_name}</strong></p>
<p>✓ Your face has been registered for future verification<br>
✓ Attendance marked<br>
✓ Classroom verification passed (✓ Bluetooth detected)<br>
✓ Face biometric captured</p>
<p><strong>{unit_code}</strong> - Week {week}</p>
<p style="font-size:13px;color:#666;">Note: Future attendances will verify your face against this registration.</p>
<p><a href='/student'><button>Return to Dashboard</button></a></p>
</div>
</div>
"""
            except sqlite3.IntegrityError:
                os.remove(face_file)
                conn.close()
                return f"""
{STYLE}
<div class="panel">
<div class="status error">
<h2>⚠️ ALREADY MARKED</h2>
<p>Attendance for <strong>{student_name}</strong> already recorded for:</p>
<p><strong>{unit_code}</strong> - Week {week}</p>
<p>You cannot mark attendance twice for the same unit and week.</p>
<p><a href='/student'><button>Return to Dashboard</button></a></p>
</div>
</div>
"""
            finally:
                conn.close()
        
        # Face verification on subsequent scans
        else:
            ref_face = cv2.imread(ref_path)
            if ref_face is None:
                os.remove(face_file)
                return f"""
{STYLE}
<div class="panel">
<div class="status error">
<h2>❌ REFERENCE FACE ERROR</h2>
<p>The stored face profile is corrupted. Please contact administrator.</p>
<p><a href='/student'><button>Return to Dashboard</button></a></p>
</div>
</div>
"""
            
            ref_face = cv2.resize(ref_face, (200, 200))
            match_score = face_recognizer.match_faces(ref_face, face_crop)
            
            if match_score < 0.65:
                os.remove(face_file)
                return f"""
{STYLE}
<div class="panel">
<div class="status error">
<h2>❌ FACE VERIFICATION FAILED - NOT AUTHORIZED</h2>
<p>Your face does not match the registered profile.</p>
<p><strong>Match Score: {match_score*100:.1f}%</strong> (Required: 65%+)</p>
<p>🚫 <strong>This is an UNAUTHORIZED ATTEMPT to mark attendance.</strong></p>
<p><strong>Possible reasons:</strong><br>
• Not the registered student<br>
• Significant change in appearance<br>
• Poor face detection quality<br>
• Different lighting/angle</p>
<p>If you are the registered student, try again with proper lighting.</p>
<p><a href='/face-scanner'><button>Try Again</button></a><br>
<a href='/student'><button>Return to Dashboard</button></a></p>
</div>
</div>
"""
            
            else:
                conn = sqlite3.connect(DB)
                c = conn.cursor()
                try:
                    c.execute(
                        "INSERT INTO attendance (unit_code, week, reg_no, student_name) VALUES (?,?,?,?)",
                        (unit_code, int(week), reg_no, student_name)
                    )
                    conn.commit()
                    os.remove(face_file)
                    return f"""
{STYLE}
<div class="panel">
<div class="status success">
<h2>✅ ATTENDANCE VERIFIED & MARKED</h2>
<p>Welcome back <strong>{student_name}</strong></p>
<p style="font-size:13px;margin:10px 0;">
  ✓ Face biometric verified<br>
  ✓ Registered student confirmed<br>
  ✓ Present in classroom (Bluetooth detected)<br>
  ✓ Attendance recorded
</p>
<p><strong>Verification Details:</strong><br>
Face Match: {match_score*100:.1f}% confidence<br>
Unit: {unit_code}<br>
Week: {week}</p>
<p><a href='/student'><button>Return to Dashboard</button></a></p>
</div>
</div>
"""
                except sqlite3.IntegrityError:
                    os.remove(face_file)
                    conn.close()
                    return f"""
{STYLE}
<div class="panel">
<div class="status error">
<h2>⚠️ ALREADY MARKED TODAY</h2>
<p>Attendance for <strong>{student_name}</strong> already recorded for:</p>
<p><strong>{unit_code}</strong> - Week {week}</p>
<p>You have already marked attendance for this unit and week.</p>
<p><a href='/student'><button>Return to Dashboard</button></a></p>
</div>
</div>
"""
                finally:
                    conn.close()
    
    except Exception as e:
        print(f"Attendance submission error: {e}")
        return f"""
{STYLE}
<div class="panel">
<div class="status error">
<h2>❌ SYSTEM ERROR</h2>
<p><strong>Error Details:</strong> {str(e)}</p>
<p>Please try again or contact your administrator if the problem persists.</p>
<p><a href='/student'><button>Return to Dashboard</button></a></p>
</div>
</div>
"""

# ==================== DOWNLOAD CSV ====================
@app.route("/download-attendance/<unit_week>")
def download_attendance(unit_week):
    if not session.get("admin"):
        return redirect("/")
    try:
        if "(Week " in unit_week:
            unit_code = unit_week.split(" (Week ")[0]
            week = int(unit_week.split(" (Week ")[1][:-1])
        else:
            return "❌ Invalid unit/week format"
        
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute(
            "SELECT student_name, reg_no, timestamp FROM attendance WHERE unit_code=? AND week=? ORDER BY timestamp",
            (unit_code, week)
        )
        students = c.fetchall()
        conn.close()
        
        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow(["Student Name", "Registration Number", "Week", "Timestamp"])
        for s, r, t in students:
            cw.writerow([s, r, week, t if t else ""])
        
        output = si.getvalue()
        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename=attendance_{unit_code}_week{week}.csv"}
        )
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ==================== BLUETOOTH STATUS ====================
@app.route("/bluetooth-status")
def bt_status():
    return jsonify({
        "enabled": bluetoothEnabled,
        "devices": bluetooth_devices
    })

# ==================== LIVE DATA ====================
@app.route("/live-data")
def live_data():
    try:
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("SELECT unit_code, week FROM attendance ORDER BY week DESC")
        data = {}
        for unit, week in c.fetchall():
            key = f"{unit} (Week {week})"
            if key not in data:
                data[key] = []
            data[key].append(1)
        conn.close()
    except Exception as e:
        print(f"Live data error: {e}")
        data = {}
    
    return jsonify({"bluetooth": bluetooth_devices, "attendance": data})

# ==================== LOGOUT ====================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ==================== RUN ====================
def open_browser():
    webbrowser.open("http://localhost:5000")

if __name__ == "__main__":
    # For online hosting, port comes from environment variable
    # For local testing, defaults to 5000
    port = int(os.environ.get('PORT', 5000))
    is_local = os.environ.get('PORT') is None  # True if running locally
    
    if is_local:
        # Only open browser when running locally
        threading.Timer(2, open_browser).start()
    
    app.run(debug=False, threaded=True, use_reloader=False, host="0.0.0.0", port=port)
