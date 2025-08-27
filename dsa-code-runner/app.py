from flask import Flask, request, jsonify, send_from_directory, render_template, session, redirect, url_for
import subprocess
import os
import tempfile
import shutil
import re
import uuid
from datetime import datetime
import base64 # Import the base64 module for image encoding

app = Flask(__name__, static_folder='public', template_folder='public')
app.config['SECRET_KEY'] = 'your-super-secret-key-change-this'

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password"

active_sessions = {}

# --- Routes (remain the same) ---

@app.route('/')
def root():
    return render_template('home.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        roll = request.form.get('roll')
        name = request.form.get('name')
        if roll and name:
            user_id = str(uuid.uuid4())
            session['user_id'] = user_id
            active_sessions[user_id] = {
                'roll': roll, 'name': name, 'ip_address': request.remote_addr,
                'login_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'status': 'active'
            }
            return redirect(url_for('ide'))
    return render_template('login.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('admin'))
        else:
            return "Invalid credentials", 401
    return render_template('admin_login.html')

@app.route('/ide')
def ide():
    if 'user_id' not in session or session['user_id'] not in active_sessions:
        return redirect(url_for('login'))
    user_details = active_sessions.get(session['user_id'])
    return render_template('index.html', user=user_details)

@app.route('/admin')
def admin():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return render_template('admin.html', sessions=active_sessions)

@app.route('/logout')
def logout():
    user_id = session.pop('user_id', None)
    if user_id in active_sessions: del active_sessions[user_id]
    return redirect(url_for('root'))

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('root'))

# --- EDITED Code Execution Logic ---

def find_java_class_name(code):
    match = re.search(r'public\s+class\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
    if match: return match.group(1)
    return None

def execute_code(code, input_data, language):
    temp_dir = tempfile.mkdtemp()
    try:
        if language == 'python':
            file_path = os.path.join(temp_dir, 'main.py')
            with open(file_path, 'w') as f: f.write(code)
            
            # For matplotlib, we need to mount the volume as read-write (:rw)
            # so the script can save the output.png file.
            docker_command = ['docker', 'run', '-i', '--rm', '--network', 'none', '--memory=512m', '--cpus=0.5', '-v', f'{temp_dir}:/usr/src/app:rw', 'python-runner', 'python', 'main.py']
            result = subprocess.run(docker_command, input=input_data, capture_output=True, text=True, timeout=20)
            
            text_output = result.stdout + result.stderr
            image_data = None
            
            # Check if the script created an output.png file
            image_path = os.path.join(temp_dir, 'output.png')
            if os.path.exists(image_path):
                with open(image_path, 'rb') as image_file:
                    # Encode the image in base64
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            return {'text': text_output, 'image': image_data}

        # C, C++, and Java logic remains the same
        elif language in ['c', 'cpp']:
            ext = 'cpp' if language == 'cpp' else 'c'
            compiler = 'g++' if language == 'cpp' else 'gcc'
            source_file = f'main.{ext}'
            file_path = os.path.join(temp_dir, source_file)
            with open(file_path, 'w') as f: f.write(code)
            compile_command = ['docker', 'run', '--rm', '--network', 'none', '--memory=512m', '--cpus=0.5', '-v', f'{temp_dir}:/usr/src/app:rw', 'cpp-runner', compiler, source_file, '-o', 'a.out']
            compile_result = subprocess.run(compile_command, capture_output=True, text=True, timeout=20)
            if compile_result.returncode != 0: return {'text': f"Compilation Error:\n{compile_result.stderr}", 'image': None}
            run_command = ['docker', 'run', '-i', '--rm', '--network', 'none', '--memory=256m', '--cpus=0.5', '--read-only', '-v', f'{temp_dir}:/usr/src/app:ro', 'cpp-runner', './a.out']
            run_result = subprocess.run(run_command, input=input_data, capture_output=True, text=True, timeout=15)
            return {'text': run_result.stdout + run_result.stderr, 'image': None}
        elif language == 'java':
            class_name = find_java_class_name(code)
            if not class_name: return {'text': "Java code error: Could not find a public class.", 'image': None}
            file_name = f"{class_name}.java"
            file_path = os.path.join(temp_dir, file_name)
            with open(file_path, 'w') as f: f.write(code)
            compile_command = ['docker', 'run', '--rm', '--network', 'none', '--memory=512m', '--cpus=1', '-v', f'{temp_dir}:/usr/src/app:rw', 'java-runner', 'javac', file_name]
            compile_result = subprocess.run(compile_command, capture_output=True, text=True, timeout=20)
            if compile_result.returncode != 0: return {'text': f"Compilation Error:\n{compile_result.stderr}", 'image': None}
            run_command = ['docker', 'run', '-i', '--rm', '--network', 'none', '--memory=256m', '--cpus=0.5', '--read-only', '-v', f'{temp_dir}:/usr/src/app:ro', 'java-runner', 'java', class_name]
            run_result = subprocess.run(run_command, input=input_data, capture_output=True, text=True, timeout=15)
            return {'text': run_result.stdout + run_result.stderr, 'image': None}
        else:
            return {'text': "Unsupported language.", 'image': None}
    finally:
        shutil.rmtree(temp_dir)

@app.route('/run', methods=['POST'])
def run_code_route():
    if 'user_id' not in session or session['user_id'] not in active_sessions:
        return jsonify({'error': 'User not authenticated.'}), 401
    try:
        data = request.get_json()
        # The output is now a dictionary, so we can jsonify it directly
        output = execute_code(data.get('code'), data.get('input_data', ''), data.get('language', 'python'))
        return jsonify(output)
    except subprocess.TimeoutExpired:
        return jsonify({'text': 'Execution timed out!', 'image': None})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'text': f'An internal server error occurred: {e}', 'image': None}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
