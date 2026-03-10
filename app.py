"""
ML-Based Course Recommendation System with Explainable AI
Main Flask Application
"""

import os
import json
import uuid
from functools import wraps
from flask import (Flask, render_template, request, redirect, url_for,
                   session, jsonify, flash, send_file)
from werkzeug.utils import secure_filename

from utils.pdf_parser import extract_marks_from_pdf, simulate_marks_from_manual_input
from utils.skill_test import get_test_questions, evaluate_submission
from utils.recommender import get_recommendation, get_model_info
from utils.data_manager import (save_user_data, save_recommendation,
                                get_all_users, get_all_recommendations, get_stats)

# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "course_rec_secret_2024")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {"pdf"}
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"  # Change in production


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────────────────────
# STUDENT ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    session.clear()
    return render_template("index.html")


@app.route("/start", methods=["GET", "POST"])
def start():
    """Step 1: Collect basic info."""
    if request.method == "POST":
        session["user_id"] = str(uuid.uuid4())[:8].upper()
        session["name"] = request.form.get("name", "Student")
        session["email"] = request.form.get("email", "")
        return redirect(url_for("upload_marksheet"))
    return render_template("start.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_marksheet():
    """Step 2: Upload PDF marksheets or enter marks manually."""
    if request.method == "POST":
        marks_data = {}

        if "manual_entry" in request.form:
            # Manual mark entry
            subjects = ["mathematics", "physics", "chemistry", "computer_science",
                        "english", "statistics", "biology", "economics"]
            subject_marks = {}
            for sem in ["sem1", "sem2", "sem3"]:
                for subj in subjects:
                    key = f"{sem}_{subj}"
                    val = request.form.get(key)
                    if val and val.strip().isdigit():
                        v = int(val)
                        if subj not in subject_marks:
                            subject_marks[subj] = []
                        subject_marks[subj].append(v)

            # Average across semesters
            avg_marks = {s: sum(v)/len(v) for s, v in subject_marks.items() if v}
            marks_data = simulate_marks_from_manual_input(avg_marks)
        else:
            # PDF upload
            uploaded_files = request.files.getlist("marksheets")
            all_marks = {}
            for f in uploaded_files[:3]:
                if f and allowed_file(f.filename):
                    fn = secure_filename(f.filename)
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], fn)
                    f.save(filepath)
                    extracted = extract_marks_from_pdf(filepath)
                    for k, v in extracted.items():
                        if k not in all_marks:
                            all_marks[k] = []
                        all_marks[k].append(v)
                    os.remove(filepath)  # Clean up after extraction

            if not all_marks:
                flash("Could not extract marks from PDFs. Please use manual entry.", "error")
                return redirect(url_for("upload_marksheet"))

            # Average across uploaded files
            marks_data = {k: round(sum(v)/len(v), 2) for k, v in all_marks.items()}

        # Store in session
        session["marks_data"] = marks_data
        session["subject_marks"] = {
            k: v for k, v in marks_data.items()
            if k in ["mathematics", "physics", "chemistry", "computer_science",
                     "english", "statistics", "biology", "economics"]
        }
        session["logical_thinking"] = marks_data.get("logical_thinking", 65)
        session["analytical_ability"] = marks_data.get("analytical_ability", 65)
        session["programming_fundamentals"] = marks_data.get("programming_fundamentals", 65)
        session["overall_average"] = marks_data.get("overall_average", 65)
        
        # Clear previous recommendation if marks changed
        session.pop("recommendation", None)

        return redirect(url_for("skill_test"))

    return render_template("upload.html")


@app.route("/skill-test", methods=["GET", "POST"])
def skill_test():
    """Step 3: Programming skill test."""
    if "overall_average" not in session:
        return redirect(url_for("start"))

    if request.method == "POST":
        language = request.form.get("language", "python")
        answers = {}
        questions = session.get("test_questions", [])

        for q in questions:
            qid = q["id"]
            answers[qid] = request.form.get(f"answer_{qid}", "")

        result = evaluate_submission(questions, answers)

        session["skill_level"] = result["skill_level"]
        session["programming_score"] = result["normalized_score"]
        session["programming_language"] = language
        session["test_result"] = result
        
        # Clear previous recommendation if test took place
        session.pop("recommendation", None)

        return redirect(url_for("interests"))

    # GET: generate questions
    language = request.args.get("lang", "python")
    questions = get_test_questions(language, n=5)
    session["test_questions"] = questions
    session["programming_language"] = language

    return render_template("skill_test.html", questions=questions, language=language)


@app.route("/interests", methods=["GET", "POST"])
def interests():
    """Step 4: Collect interests and career goals."""
    if "skill_level" not in session:
        return redirect(url_for("start"))

    INTEREST_OPTIONS = [
        ("data_science", "Data Science"),
        ("web_development", "Web Development"),
        ("mobile_development", "Mobile Development"),
        ("ai_ml", "Artificial Intelligence / ML"),
        ("cybersecurity", "Cybersecurity"),
        ("cloud_computing", "Cloud Computing"),
        ("game_development", "Game Development"),
        ("design", "UI/UX Design"),
        ("database", "Database Engineering"),
        ("devops", "DevOps / SRE"),
        ("blockchain", "Blockchain / Web3"),
        ("management", "Project Management")
    ]

    CAREER_OPTIONS = [
        "Data Scientist", "ML Engineer", "AI Researcher", "Web Developer",
        "Frontend Developer", "Backend Developer", "Full Stack Developer",
        "Mobile Developer", "Cloud Engineer", "DevOps Engineer",
        "Security Analyst", "Data Analyst", "BI Analyst", "Database Administrator",
        "Software Engineer", "Software Architect", "Game Developer",
        "UI/UX Designer", "Project Manager", "Blockchain Developer",
        "iOS Developer", "Android Developer", "NLP Engineer",
        "Computer Vision Engineer", "Statistician", "Penetration Tester"
    ]

    if request.method == "POST":
        session["interest"] = request.form.get("interest", "data_science")
        session["career_goal"] = request.form.get("career_goal", "Data Scientist")
        
        # Clear previous recommendation if interests changed
        session.pop("recommendation", None)
        return redirect(url_for("recommend"))

    return render_template("interests.html",
                           interest_options=INTEREST_OPTIONS,
                           career_options=CAREER_OPTIONS,
                           skill_level=session.get("skill_level", "beginner"),
                           avg=session.get("overall_average", 65))


@app.route("/recommend")
def recommend():
    """Step 5: Generate and display recommendation."""
    required_keys = ["skill_level", "interest", "career_goal", "overall_average"]
    if not all(k in session for k in required_keys):
        return redirect(url_for("start"))

    user_data = {
        "user_id": session.get("user_id"),
        "name": session.get("name", "Student"),
        "email": session.get("email", ""),
        "subject_marks": session.get("subject_marks", {}),
        "overall_average": session.get("overall_average", 65),
        "logical_thinking": session.get("logical_thinking", 65),
        "analytical_ability": session.get("analytical_ability", 65),
        "programming_fundamentals": session.get("programming_fundamentals", 65),
        "programming_score": session.get("programming_score", 50),
        "skill_level": session.get("skill_level", "beginner"),
        "programming_language": session.get("programming_language", "python"),
        "interest": session.get("interest", "data_science"),
        "career_goal": session.get("career_goal", "Data Scientist")
    }

    # If recommendation already exists in session, don't re-save (prevents duplicates on refresh)
    if "recommendation" in session:
        result = session["recommendation"]
    else:
        try:
            result = get_recommendation(user_data)
            user_id = save_user_data(user_data)
            rec_id = save_recommendation(user_id, result)
            session["recommendation"] = result
            session["rec_id"] = rec_id
        except FileNotFoundError as e:
            flash(str(e), "error")
            return redirect(url_for("index"))
        except Exception as e:
            flash(f"Recommendation error: {str(e)}", "error")
            return redirect(url_for("interests"))

    return render_template("recommendation.html",
                           result=result,
                           user_data=user_data,
                           test_result=session.get("test_result", {}))


# ─────────────────────────────────────────────────────────────
# ADMIN ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("admin_dashboard"))
        flash("Invalid credentials.", "error")
    return render_template("admin/login.html")


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    stats = get_stats()
    try:
        model_info = get_model_info()
    except Exception:
        model_info = {}
    return render_template("admin/dashboard.html", stats=stats, model_info=model_info)


@app.route("/admin/users")
@admin_required
def admin_users():
    users = get_all_users()
    return render_template("admin/users.html", users=users)


@app.route("/admin/recommendations")
@admin_required
def admin_recommendations():
    recs = get_all_recommendations()
    return render_template("admin/recommendations.html", recs=recs)


@app.route("/admin/export/users")
@admin_required
def export_users():
    from utils.data_manager import USERS_CSV
    return send_file(USERS_CSV, as_attachment=True, download_name="users.csv")


@app.route("/admin/export/recommendations")
@admin_required
def export_recommendations():
    from utils.data_manager import RECOMMENDATIONS_CSV
    return send_file(RECOMMENDATIONS_CSV, as_attachment=True, download_name="recommendations.csv")


@app.route("/admin/model-info")
@admin_required
def admin_model_info():
    try:
        info = get_model_info()
    except Exception as e:
        info = {"error": str(e)}
    return render_template("admin/model_info.html", info=info)


# ─────────────────────────────────────────────────────────────
# API ENDPOINTS (for AJAX)
# ─────────────────────────────────────────────────────────────

@app.route("/api/questions/<language>")
def api_questions(language):
    questions = get_test_questions(language, n=5)
    session["test_questions"] = questions
    session["programming_language"] = language
    return jsonify(questions)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
