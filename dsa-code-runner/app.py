from flask import Flask, request, jsonify, send_from_directory, render_template, session, redirect, url_for
import subprocess
import os
import tempfile
import shutil
import re
import uuid
from datetime import datetime

app = Flask(__name__, static_folder='public', template_folder='public')
app.config['SECRET_KEY'] = 'your-super-secret-key-change-this'

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password"

active_sessions = {}

# --- Routes ---

@app.route('/')
def root():
    return render_template('home.html')

@app.route('/docs')
def docs():
    # New route to serve the documentation page
    return render_template('docs.html')

# --- Student Login ---
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

# --- Admin Login ---
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
    if user_id in active_sessions:
        del active_sessions[user_id]
    return redirect(url_for('root'))

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('root'))

# --- Code Execution Logic (remains the same) ---
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
            docker_command = ['docker', 'run', '-i', '--rm', '--network', 'none', '--memory=256m', '--cpus=0.5', '--read-only', '-v', f'{temp_dir}:/usr/src/app:ro', 'python-runner', 'python', 'main.py']
            result = subprocess.run(docker_command, input=input_data, capture_output=True, text=True, timeout=15)
            return result.stdout + result.stderr
        elif language in ['c', 'cpp']:
            ext = 'cpp' if language == 'cpp' else 'c'
            compiler = 'g++' if language == 'cpp' else 'gcc'
            source_file = f'main.{ext}'
            file_path = os.path.join(temp_dir, source_file)
            with open(file_path, 'w') as f: f.write(code)
            compile_command = ['docker', 'run', '--rm', '--network', 'none', '--memory=512m', '--cpus=0.5', '-v', f'{temp_dir}:/usr/src/app:rw', 'cpp-runner', compiler, source_file, '-o', 'a.out']
            compile_result = subprocess.run(compile_command, capture_output=True, text=True, timeout=20)
            if compile_result.returncode != 0: return f"Compilation Error:\n{compile_result.stderr}"
            run_command = ['docker', 'run', '-i', '--rm', '--network', 'none', '--memory=256m', '--cpus=0.5', '--read-only', '-v', f'{temp_dir}:/usr/src/app:ro', 'cpp-runner', './a.out']
            run_result = subprocess.run(run_command, input=input_data, capture_output=True, text=True, timeout=15)
            return run_result.stdout + run_result.stderr
        elif language == 'java':
            class_name = find_java_class_name(code)
            if not class_name: return "Java code error: Could not find a public class."
            file_name = f"{class_name}.java"
            file_path = os.path.join(temp_dir, file_name)
            with open(file_path, 'w') as f: f.write(code)
            compile_command = ['docker', 'run', '--rm', '--network', 'none', '--memory=512m', '--cpus=1', '-v', f'{temp_dir}:/usr/src/app:rw', 'java-runner', 'javac', file_name]
            compile_result = subprocess.run(compile_command, capture_output=True, text=True, timeout=20)
            if compile_result.returncode != 0: return f"Compilation Error:\n{compile_result.stderr}"
            run_command = ['docker', 'run', '-i', '--rm', '--network', 'none', '--memory=256m', '--cpus=0.5', '--read-only', '-v', f'{temp_dir}:/usr/src/app:ro', 'java-runner', 'java', class_name]
            run_result = subprocess.run(run_command, input=input_data, capture_output=True, text=True, timeout=15)
            return run_result.stdout + run_result.stderr
        else:
            return "Unsupported language."
    finally:
        shutil.rmtree(temp_dir)

@app.route('/run', methods=['POST'])
def run_code_route():
    if 'user_id' not in session or session['user_id'] not in active_sessions:
        return jsonify({'error': 'User not authenticated.'}), 401
    try:
        data = request.get_json()
        output = execute_code(data.get('code'), data.get('input_data', ''), data.get('language', 'python'))
        return jsonify({'output': output})
    except subprocess.TimeoutExpired:
        return jsonify({'output': 'Execution timed out!'})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': f'An internal server error occurred: {e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)