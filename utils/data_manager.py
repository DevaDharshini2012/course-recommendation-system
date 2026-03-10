"""
Data Manager - Handles CSV-based user data and recommendation storage.
Admin-only access enforced via Flask routes.
"""

import os
import csv
import json
import uuid
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

USERS_CSV = os.path.join(DATA_DIR, "users.csv")
RECOMMENDATIONS_CSV = os.path.join(DATA_DIR, "recommendations.csv")
SESSIONS_CSV = os.path.join(DATA_DIR, "sessions.csv")

# ─────────────────────────────────────────────────────────────
# CSV Schema Definitions
# ─────────────────────────────────────────────────────────────
USERS_FIELDS = [
    "user_id", "name", "email", "timestamp",
    "math_mark", "physics_mark", "chemistry_mark", "cs_mark",
    "english_mark", "statistics_mark", "biology_mark", "economics_mark",
    "overall_average", "logical_thinking", "analytical_ability",
    "programming_fundamentals", "skill_level", "programming_language",
    "programming_score", "interest", "career_goal"
]

RECOMMENDATIONS_FIELDS = [
    "rec_id", "user_id", "timestamp",
    "recommended_course_id", "recommended_course_name",
    "confidence_percent", "category", "level",
    "alternative_1", "alternative_2",
    "key_reason_1", "key_reason_2", "key_reason_3",
    "top_feature_1", "top_feature_2", "top_feature_3",
    "model_used"
]

SESSIONS_FIELDS = [
    "session_id", "user_id", "start_time", "end_time",
    "phase_completed", "pdf_uploaded", "test_completed", "recommendation_generated"
]


def _ensure_csv(filepath: str, fields: list):
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()


def save_user_data(user_data: dict) -> str:
    """Save user profile data to CSV. Returns generated user_id."""
    _ensure_csv(USERS_CSV, USERS_FIELDS)

    user_id = user_data.get("user_id") or str(uuid.uuid4())[:8].upper()
    subject_marks = user_data.get("subject_marks", {})

    row = {
        "user_id": user_id,
        "name": user_data.get("name", "Anonymous"),
        "email": user_data.get("email", ""),
        "timestamp": datetime.now().isoformat(),
        "math_mark": subject_marks.get("mathematics", ""),
        "physics_mark": subject_marks.get("physics", ""),
        "chemistry_mark": subject_marks.get("chemistry", ""),
        "cs_mark": subject_marks.get("computer_science", ""),
        "english_mark": subject_marks.get("english", ""),
        "statistics_mark": subject_marks.get("statistics", ""),
        "biology_mark": subject_marks.get("biology", ""),
        "economics_mark": subject_marks.get("economics", ""),
        "overall_average": user_data.get("overall_average", ""),
        "logical_thinking": user_data.get("logical_thinking", ""),
        "analytical_ability": user_data.get("analytical_ability", ""),
        "programming_fundamentals": user_data.get("programming_fundamentals", ""),
        "skill_level": user_data.get("skill_level", ""),
        "programming_language": user_data.get("programming_language", ""),
        "programming_score": user_data.get("programming_score", ""),
        "interest": user_data.get("interest", ""),
        "career_goal": user_data.get("career_goal", "")
    }

    with open(USERS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=USERS_FIELDS)
        writer.writerow(row)

    return user_id


def save_recommendation(user_id: str, rec_result: dict) -> str:
    """Save recommendation result to CSV. Returns rec_id."""
    _ensure_csv(RECOMMENDATIONS_CSV, RECOMMENDATIONS_FIELDS)

    rec_id = str(uuid.uuid4())[:8].upper()
    course = rec_result.get("recommended_course", {})
    alts = rec_result.get("alternative_courses", [])
    reasons = rec_result.get("explanation", {}).get("natural_language_reasons", [])
    top_feats = rec_result.get("explanation", {}).get("top_contributing_features", [])

    row = {
        "rec_id": rec_id,
        "user_id": user_id,
        "timestamp": rec_result.get("timestamp", datetime.now().isoformat()),
        "recommended_course_id": course.get("course_id", ""),
        "recommended_course_name": course.get("course_name", ""),
        "confidence_percent": rec_result.get("confidence", ""),
        "category": course.get("category", ""),
        "level": course.get("level", ""),
        "alternative_1": alts[0]["course_name"] if len(alts) > 0 else "",
        "alternative_2": alts[1]["course_name"] if len(alts) > 1 else "",
        "key_reason_1": reasons[0][:200] if len(reasons) > 0 else "",
        "key_reason_2": reasons[1][:200] if len(reasons) > 1 else "",
        "key_reason_3": reasons[2][:200] if len(reasons) > 2 else "",
        "top_feature_1": top_feats[0]["feature"] if len(top_feats) > 0 else "",
        "top_feature_2": top_feats[1]["feature"] if len(top_feats) > 1 else "",
        "top_feature_3": top_feats[2]["feature"] if len(top_feats) > 2 else "",
        "model_used": "Random Forest + Decision Tree XAI"
    }

    with open(RECOMMENDATIONS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RECOMMENDATIONS_FIELDS)
        writer.writerow(row)

    return rec_id


def get_all_users() -> list:
    """Admin only: Return all user records."""
    _ensure_csv(USERS_CSV, USERS_FIELDS)
    rows = []
    with open(USERS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def get_all_recommendations() -> list:
    """Admin only: Return all recommendation records."""
    _ensure_csv(RECOMMENDATIONS_CSV, RECOMMENDATIONS_FIELDS)
    rows = []
    with open(RECOMMENDATIONS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def get_stats() -> dict:
    """Admin dashboard stats."""
    users = get_all_users()
    recs = get_all_recommendations()

    # Course distribution
    course_counts = {}
    for r in recs:
        name = r.get("recommended_course_name", "Unknown")
        course_counts[name] = course_counts.get(name, 0) + 1

    # Skill level distribution
    skill_counts = {}
    for u in users:
        level = u.get("skill_level", "unknown")
        skill_counts[level] = skill_counts.get(level, 0) + 1

    # Interest distribution
    interest_counts = {}
    for u in users:
        interest = u.get("interest", "unknown")
        interest_counts[interest] = interest_counts.get(interest, 0) + 1

    return {
        "total_users": len(users),
        "total_recommendations": len(recs),
        "course_distribution": dict(sorted(course_counts.items(), key=lambda x: x[1], reverse=True)),
        "skill_distribution": skill_counts,
        "interest_distribution": dict(sorted(interest_counts.items(), key=lambda x: x[1], reverse=True))
    }
