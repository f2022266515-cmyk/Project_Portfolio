import os
import google.generativeai as genai
from flask import Flask, jsonify, render_template, request, session, redirect
from functools import wraps
from flask_socketio import SocketIO, emit
from dms_database import DMSDatabase
import requests as req_lib
from datetime import datetime, timedelta, timezone
import re

app = Flask(__name__,
            static_folder='static',
            template_folder='templates',
            static_url_path='/static')

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'development-fallback-key-2026')
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

socketio = SocketIO(app, cors_allowed_origins="*")

db = DMSDatabase()

# ============================================================
# AUTH DECORATOR
# ============================================================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'admin_email' not in session:
            if request.is_json or request.path.startswith('/api/'):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect('/products')
        return f(*args, **kwargs)
    return decorated

# ============================================================
# PAGE ROUTES
# ============================================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/products')
def products():
    return render_template('products.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/records')
@login_required
def records():
    return render_template('records.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/logs')
@login_required
def logs():
    return render_template('logs.html')

# ============================================================
# DIAGNOSTIC API
# ============================================================
@app.route('/api/health')
def health_check():
    try:
        if db.client:
            db.client.admin.command('ping')
            return jsonify({"status": "healthy", "database": "connected"}), 200
        else:
            return jsonify({"status": "unhealthy", "database": "disconnected", "error": "Client not initialized"}), 500
    except Exception as e:
        return jsonify({"status": "unhealthy", "database": "disconnected", "error": str(e)}), 500

# ============================================================
# AUTH API
# ============================================================
@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.json
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')

        user = db.verify_user(email, password)
        if user:
            session['admin_email'] = email
            session['admin_name'] = user.get('name', email.split('@')[0].title())
            session['admin_drivers'] = user.get('assigned_drivers', [])
            return jsonify({"success": True, "name": session['admin_name']})

        return jsonify({"success": False, "error": "Invalid email or password"}), 401
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({"success": True})

@app.route('/api/me')
def api_me():
    if 'admin_email' not in session:
        return jsonify({"logged_in": False})

    email = session['admin_email']
    user_data = db.get_user(email)

    if user_data:
        raw_devices = user_data.get('fleet_devices', [])
        dynamic_devices = []
        now = datetime.now(timezone.utc)

        for dev in raw_devices:
            last_seen = dev.get('last_seen')
            is_online = False
            if last_seen:
                if isinstance(last_seen, datetime):
                    if last_seen.tzinfo is None:
                        last_seen = last_seen.replace(tzinfo=timezone.utc)
                    if (now - last_seen).total_seconds() < 90:
                        is_online = True

            dynamic_devices.append({
                "unit_id": dev.get("unit_id"),
                "assigned_driver": dev.get("assigned_driver"),
                "car_model": dev.get("car_model"),
                "status": "Online" if is_online else "Offline"
            })

        return jsonify({
            "logged_in": True,
            "email": email,
            "name": session.get('admin_name', user_data.get('name')),
            "drivers": session.get('admin_drivers', user_data.get('assigned_drivers', [])),
            "devices": dynamic_devices
        })
    return jsonify({"logged_in": False})

@app.route('/api/change_password', methods=['POST'])
@login_required
def change_password():
    try:
        data = request.json
        current_pass = data.get('current_password', '')
        new_pass = data.get('new_password', '')

        if not new_pass or len(new_pass) < 8:
            return jsonify({"success": False, "error": "Password must be at least 8 characters long."}), 400

        email = session['admin_email']
        if not db.verify_user(email, current_pass):
            return jsonify({"success": False, "error": "Current password is incorrect"}), 401

        db.update_password(email, new_pass)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/heartbeat', methods=['POST'])
def device_heartbeat():
    data = request.get_json()
    unit_id = data.get('unit_id') if data else None
    if unit_id and db.update_device_heartbeat(unit_id):
        return jsonify({"success": True}), 200
    return jsonify({"success": False, "error": "Unit ID not found"}), 404

@app.route('/api/events')
@login_required
def get_events():
    email = session.get('admin_email')
    user_data = db.get_user(email)
    drivers = user_data.get('assigned_drivers', []) if user_data else []
    req_type = request.args.get('type', 'all')

    events = db.get_events_for_drivers(drivers, limit=500 if req_type == 'all' else 50)

    if req_type == 'live':
        filtered_events = []
        twelve_hours_ago = datetime.now() - timedelta(hours=12)

        for e in events:
            ts = e.get('timestamp')
            if not ts:
                continue
            try:
                if isinstance(ts, str):
                    clean_ts = ts.replace('T', ' ')
                    event_time = datetime.strptime(clean_ts, "%Y-%m-%d %H:%M:%S")
                else:
                    event_time = ts
                if event_time >= twelve_hours_ago:
                    filtered_events.append(e)
            except Exception as ex:
                print(f"Timestamp parse error: {ex}")
                filtered_events.append(e)

        return jsonify(filtered_events)

    return jsonify(events)

@app.route('/api/stats')
@login_required
def get_stats():
    try:
        email = session.get('admin_email')
        user_data = db.get_user(email)
        drivers = user_data.get('assigned_drivers', []) if user_data else []

        events = db.get_events_for_drivers(drivers, limit=500)
        total = len(events)
        sleep   = sum(1 for e in events if any(w in str(e.get('event_type', '')).lower() for w in ['sleep', 'drowsy', 'yawn']))
        smoking = sum(1 for e in events if any(w in str(e.get('event_type', '')).lower() for w in ['smok', 'vape', 'cig']))

        calculated_score = 100 - (total * 1)
        final_score = max(0, calculated_score)

        return jsonify({
            "total_violations": total,
            "sleep_alerts": sleep,
            "smoking_cases": smoking,
            "safety_score": final_score
        })
    except Exception as e:
        return jsonify({"error": str(e), "total_violations": 0})

# ============================================================
# NEW: WEBSOCKET (REAL-TIME STREAMING) API
# ============================================================
@socketio.on('connect')
def handle_connect():
    print("Client connected for real-time DMS streaming.")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected.")

@socketio.on('stream_data')
def handle_stream_data(data):
    """
    Receives real-time bounding boxes, EAR/MAR metrics, or alert flags 
    from the local driver interface and broadcasts them to the admin dashboard.
    """
    unit_id = data.get('unit_id')
    alert_type = data.get('alert_type')
    
    if alert_type:
        # Broadcast critical alerts instantly without waiting for HTTP polling
        emit('dms_alert_triggered', {'unit_id': unit_id, 'alert': alert_type}, broadcast=True)

# ============================================================
# CHAT API
# ============================================================
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json or {}
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"reply": "Please type a message."})

        API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

        if not API_KEY:
            return jsonify({"reply": "⚠️ AI not configured. Add GEMINI_API_KEY in Render → Environment."})

        genai.configure(api_key=API_KEY)
        
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=(
                "You are a DMS (Driver Monitoring System) Safety Assistant. "
                "Answer questions about driver safety, drowsiness detection, "
                "smoking detection, seatbelt violations, and fleet monitoring. "
                "Keep answers under 80 words and be helpful."
            )
        )

        response = model.generate_content(message)
        
        if response and response.text:
            return jsonify({"reply": response.text})
        else:
            return jsonify({"reply": "⚠️ Empty response from AI. Please try again."})

    except Exception as e:
        return jsonify({"reply": f"⚠️ Chat system error: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # ✨ NEW: Run via socketio instead of standard app.run to enable WebSocket communication
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
