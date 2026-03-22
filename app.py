# Smart Lab Submission and Plagiarism Checker System
# Main application file: app.py

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import sqlite3
from datetime import datetime
import hashlib
import difflib
import re
from pathlib import Path
import zipfile
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.txt', '.py', '.cpp', '.c', '.java', '.js', '.html', '.css', '.md', '.pdf'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

class DatabaseManager:
    def __init__(self, db_name='lab_submissions.db'):
        self.db_name = db_name
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Create submissions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                student_id TEXT NOT NULL,
                assignment_name TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                submission_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                plagiarism_score REAL DEFAULT 0.0,
                status TEXT DEFAULT 'submitted'
            )
        ''')
        
        # Create plagiarism_results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plagiarism_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                submission_id INTEGER,
                compared_with INTEGER,
                similarity_score REAL,
                FOREIGN KEY (submission_id) REFERENCES submissions (id),
                FOREIGN KEY (compared_with) REFERENCES submissions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_submission(self, student_name, student_id, assignment_name, filename, file_path, file_hash):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO submissions (student_name, student_id, assignment_name, filename, file_path, file_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (student_name, student_id, assignment_name, filename, file_path, file_hash))
        submission_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return submission_id
    
    def get_submissions(self, assignment_name=None):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        if assignment_name:
            cursor.execute('SELECT * FROM submissions WHERE assignment_name = ? ORDER BY submission_time DESC', (assignment_name,))
        else:
            cursor.execute('SELECT * FROM submissions ORDER BY submission_time DESC')
        submissions = cursor.fetchall()
        conn.close()
        return submissions
    
    def update_plagiarism_score(self, submission_id, score):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('UPDATE submissions SET plagiarism_score = ? WHERE id = ?', (score, submission_id))
        conn.commit()
        conn.close()

class PlagiarismChecker:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        )
    
    def read_file_content(self, file_path):
        """Read content from various file types"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                return content
            except:
                return ""
        except:
            return ""
    
    def preprocess_code(self, content):
        """Preprocess code by removing comments and normalizing"""
        # Remove single-line comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        content = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
        content = re.sub(r"'''.*?'''", '', content, flags=re.DOTALL)
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Extract function/variable names and structure
        tokens = word_tokenize(content.lower())
        return ' '.join(tokens)
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        try:
            processed_text1 = self.preprocess_code(text1)
            processed_text2 = self.preprocess_code(text2)
            
            if not processed_text1.strip() or not processed_text2.strip():
                return 0.0
            
            # Use TF-IDF vectorization
            tfidf_matrix = self.vectorizer.fit_transform([processed_text1, processed_text2])
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix[0][1] * 100  # Convert to percentage
        except:
            return 0.0
    
    def check_plagiarism(self, new_file_path, existing_files):
        """Check plagiarism against existing submissions"""
        new_content = self.read_file_content(new_file_path)
        if not new_content:
            return []
        
        results = []
        for file_path, submission_info in existing_files:
            existing_content = self.read_file_content(file_path)
            if existing_content:
                similarity = self.calculate_similarity(new_content, existing_content)
                if similarity > 10:  # Only report similarities above 10%
                    results.append({
                        'file': submission_info,
                        'similarity': similarity
                    })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)

# Initialize components
db_manager = DatabaseManager()
plagiarism_checker = PlagiarismChecker()

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def calculate_file_hash(file_path):
    """Calculate MD5 hash of file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit_assignment():
    if request.method == 'POST':
        # Get form data
        student_name = request.form['student_name']
        student_id = request.form['student_id']
        assignment_name = request.form['assignment_name']
        
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{student_id}_{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            file.save(file_path)
            
            # Calculate file hash
            file_hash = calculate_file_hash(file_path)
            
            # Check for duplicate submissions
            existing_submissions = db_manager.get_submissions(assignment_name)
            duplicate_found = False
            for submission in existing_submissions:
                if submission[6] == file_hash:  # file_hash column
                    duplicate_found = True
                    flash(f'This file has already been submitted by {submission[1]} at {submission[7]}')
                    os.remove(file_path)
                    break
            
            if not duplicate_found:
                # Add to database
                submission_id = db_manager.add_submission(
                    student_name, student_id, assignment_name, filename, file_path, file_hash
                )
                
                # Check for plagiarism
                existing_files = []
                for submission in existing_submissions:
                    if os.path.exists(submission[5]):  # file_path column
                        existing_files.append((submission[5], {
                            'student_name': submission[1],
                            'student_id': submission[2],
                            'filename': submission[4]
                        }))
                
                plagiarism_results = plagiarism_checker.check_plagiarism(file_path, existing_files)
                
                # Calculate overall plagiarism score
                max_similarity = max([r['similarity'] for r in plagiarism_results]) if plagiarism_results else 0
                db_manager.update_plagiarism_score(submission_id, max_similarity)
                
                flash(f'Assignment submitted successfully! Plagiarism score: {max_similarity:.1f}%')
                return redirect(url_for('submission_details', submission_id=submission_id))
        else:
            flash('Invalid file type. Allowed types: .txt, .py, .cpp, .c, .java, .js, .html, .css, .md, .pdf')
    
    return render_template('submit.html')

@app.route('/submissions')
def view_submissions():
    assignment_name = request.args.get('assignment')
    submissions = db_manager.get_submissions(assignment_name)
    
    # Get unique assignment names for filter
    all_submissions = db_manager.get_submissions()
    assignment_names = list(set([s[3] for s in all_submissions]))
    
    return render_template('submissions.html', 
                         submissions=submissions, 
                         assignment_names=assignment_names,
                         selected_assignment=assignment_name)

@app.route('/submission/<int:submission_id>')
def submission_details(submission_id):
    submissions = db_manager.get_submissions()
    submission = None
    for s in submissions:
        if s[0] == submission_id:
            submission = s
            break
    
    if not submission:
        flash('Submission not found')
        return redirect(url_for('view_submissions'))
    
    # Get plagiarism results
    existing_submissions = db_manager.get_submissions(submission[3])  # assignment_name
    existing_files = []
    for s in existing_submissions:
        if s[0] != submission_id and os.path.exists(s[5]):
            existing_files.append((s[5], {
                'student_name': s[1],
                'student_id': s[2],
                'filename': s[4],
                'submission_id': s[0]
            }))
    
    plagiarism_results = plagiarism_checker.check_plagiarism(submission[5], existing_files)
    
    return render_template('submission_details.html', 
                         submission=submission, 
                         plagiarism_results=plagiarism_results)

@app.route('/download/<int:submission_id>')
def download_file(submission_id):
    submissions = db_manager.get_submissions()
    for submission in submissions:
        if submission[0] == submission_id:
            if os.path.exists(submission[5]):
                return send_file(submission[5], as_attachment=True, 
                               download_name=submission[4])
            else:
                flash('File not found')
                break
    return redirect(url_for('view_submissions'))

@app.route('/analytics')
def analytics():
    submissions = db_manager.get_submissions()
    
    # Calculate statistics
    total_submissions = len(submissions)
    high_plagiarism = len([s for s in submissions if s[8] > 50])  # plagiarism_score > 50%
    medium_plagiarism = len([s for s in submissions if 20 < s[8] <= 50])
    low_plagiarism = len([s for s in submissions if s[8] <= 20])
    
    # Assignment-wise statistics
    assignment_stats = {}
    for submission in submissions:
        assignment = submission[3]
        if assignment not in assignment_stats:
            assignment_stats[assignment] = {'total': 0, 'avg_plagiarism': 0, 'scores': []}
        assignment_stats[assignment]['total'] += 1
        assignment_stats[assignment]['scores'].append(submission[8])
    
    for assignment in assignment_stats:
        scores = assignment_stats[assignment]['scores']
        assignment_stats[assignment]['avg_plagiarism'] = sum(scores) / len(scores) if scores else 0
    
    return render_template('analytics.html',
                         total_submissions=total_submissions,
                         high_plagiarism=high_plagiarism,
                         medium_plagiarism=medium_plagiarism,
                         low_plagiarism=low_plagiarism,
                         assignment_stats=assignment_stats)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)