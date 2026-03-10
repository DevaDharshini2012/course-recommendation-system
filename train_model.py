"""
ML Model Training Script for Course Recommendation System
Run this script once to train and save the model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import json

# ─────────────────────────────────────────────────────────────
# 1. Load course dataset
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COURSES_PATH = os.path.join(BASE_DIR, "data", "courses.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

courses_df = pd.read_csv(COURSES_PATH)

# ─────────────────────────────────────────────────────────────
# 2. Define feature schema
# ─────────────────────────────────────────────────────────────
ALL_SUBJECTS = [
    "mathematics", "physics", "chemistry", "computer_science",
    "english", "statistics", "biology", "economics"
]

ALL_SKILLS = [
    "logical_thinking", "analytical_ability", "programming_basics",
    "programming_intermediate", "programming_advanced", "mathematics",
    "statistics", "creativity", "networking_basics", "system_design",
    "communication", "leadership", "business_acumen", "cryptography",
    "linguistics", "networking"
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

INTEREST_OPTIONS = [
    "data_science", "web_development", "mobile_development", "ai_ml",
    "cybersecurity", "cloud_computing", "game_development", "design",
    "database", "devops", "blockchain", "management"
]

SKILL_LEVELS = ["beginner", "intermediate", "advanced"]


# ─────────────────────────────────────────────────────────────
# 3. Generate synthetic training data
# ─────────────────────────────────────────────────────────────
def generate_training_data(n_samples=3000):
    np.random.seed(42)
    records = []

    for _ in range(n_samples):
        # Random subject marks (0-100) for last 3 semesters
        marks = {subj: np.random.randint(40, 100) for subj in ALL_SUBJECTS}
        avg_marks = np.mean(list(marks.values()))

        # Derived strengths from marks
        cs_mark = marks.get("computer_science", 60)
        math_mark = marks.get("mathematics", 60)
        stat_mark = marks.get("statistics", 60)

        logical_thinking = min(100, (cs_mark * 0.5 + math_mark * 0.3 + avg_marks * 0.2))
        analytical_ability = min(100, (math_mark * 0.4 + stat_mark * 0.4 + avg_marks * 0.2))
        prog_basics = min(100, cs_mark * 0.7 + math_mark * 0.3)

        # Programming skill level
        prog_score = np.random.randint(0, 100)
        if prog_score < 35:
            skill_level = "beginner"
        elif prog_score < 70:
            skill_level = "intermediate"
        else:
            skill_level = "advanced"

        # Random interest & career
        interest = np.random.choice(INTEREST_OPTIONS)
        career = np.random.choice(CAREER_OPTIONS)

        # Rule-based label (course recommendation) for training
        course_id = _assign_course(
            interest, career, skill_level,
            logical_thinking, analytical_ability, prog_basics, avg_marks
        )

        # ADD NOISE: 5% chance of a random course assignment
        if np.random.random() < 0.05:
            course_id = np.random.choice(courses_df["course_id"].values)

        record = {
            **{f"avg_{s}": marks[s] for s in ALL_SUBJECTS},
            "avg_all_marks": avg_marks,
            "logical_thinking": logical_thinking,
            "analytical_ability": analytical_ability,
            "programming_score": prog_basics,
            "skill_level": skill_level,
            "interest": interest,
            "career_goal": career,
            "recommended_course_id": course_id
        }
        records.append(record)

    return pd.DataFrame(records)


def _assign_course(interest, career, skill_level, logical, analytical, prog, avg):
    """Rule-based assignment for generating labeled training data."""
    rules = {
        ("data_science", "beginner"): "C001",
        ("data_science", "intermediate"): "C002",
        ("data_science", "advanced"): "C003",
        ("ai_ml", "beginner"): "C002",
        ("ai_ml", "intermediate"): "C003",
        ("ai_ml", "advanced"): "C019",
        ("web_development", "beginner"): "C004",
        ("web_development", "intermediate"): "C005",
        ("web_development", "advanced"): "C006",
        ("cloud_computing", "beginner"): "C007",
        ("cloud_computing", "intermediate"): "C007",
        ("cloud_computing", "advanced"): "C008",
        ("cybersecurity", "beginner"): "C009",
        ("cybersecurity", "intermediate"): "C009",
        ("cybersecurity", "advanced"): "C010",
        ("database", "beginner"): "C011",
        ("database", "intermediate"): "C012",
        ("database", "advanced"): "C017",
        ("mobile_development", "beginner"): "C013",
        ("mobile_development", "intermediate"): "C015",
        ("mobile_development", "advanced"): "C014",
        ("game_development", "beginner"): "C004",
        ("game_development", "intermediate"): "C023",
        ("game_development", "advanced"): "C023",
        ("design", "beginner"): "C024",
        ("design", "intermediate"): "C024",
        ("design", "advanced"): "C023",
        ("devops", "beginner"): "C007",
        ("devops", "intermediate"): "C008",
        ("devops", "advanced"): "C017",
        ("blockchain", "beginner"): "C011",
        ("blockchain", "intermediate"): "C016",
        ("blockchain", "advanced"): "C022",
        ("management", "beginner"): "C025",
        ("management", "intermediate"): "C025",
        ("management", "advanced"): "C025",
    }

    # Career overrides
    career_overrides = {
        "Data Scientist": "C002", "ML Engineer": "C002", "AI Researcher": "C003",
        "NLP Engineer": "C019", "Computer Vision Engineer": "C018",
        "Penetration Tester": "C010", "Blockchain Developer": "C022",
        "Statistician": "C021", "BI Analyst": "C020"
    }

    if career in career_overrides and skill_level == "advanced":
        return career_overrides[career]

    key = (interest, skill_level)
    return rules.get(key, "C001")


# ─────────────────────────────────────────────────────────────
# 4. Feature engineering
# ─────────────────────────────────────────────────────────────
def encode_features(df, fit=True, encoders=None):
    """Encode categorical features."""
    if encoders is None:
        encoders = {}

    cat_cols = ["skill_level", "interest", "career_goal"]
    df_enc = df.copy()

    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df_enc[col + "_enc"] = le.fit_transform(df_enc[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df_enc[col + "_enc"] = le.transform(df_enc[col])

    return df_enc, encoders


# ─────────────────────────────────────────────────────────────
# 5. Train model
# ─────────────────────────────────────────────────────────────
def train():
    print("Generating synthetic training data...")
    df = generate_training_data(3000)

    df_enc, encoders = encode_features(df, fit=True)

    feature_cols = (
        [f"avg_{s}" for s in ALL_SUBJECTS] +
        ["avg_all_marks", "logical_thinking", "analytical_ability",
         "programming_score", "skill_level_enc", "interest_enc", "career_goal_enc"]
    )

    X = df_enc[feature_cols]
    y = df_enc["recommended_course_id"]

    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y)
    encoders["target"] = le_target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    print("Training Decision Tree model (for explainability)...")
    dt_model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=15,
        random_state=42
    )
    dt_model.fit(X_train, y_train)

    # Evaluation
    y_pred_rf = rf_model.predict(X_test)
    y_pred_dt = dt_model.predict(X_test)

    rf_acc = accuracy_score(y_test, y_pred_rf)
    dt_acc = accuracy_score(y_test, y_pred_dt)

    print(f"\nRandom Forest Accuracy: {rf_acc:.4f}")
    print(f"Decision Tree Accuracy: {dt_acc:.4f}")

    cv_scores = cross_val_score(rf_model, X, y_enc, cv=5, scoring='accuracy')
    print(f"RF Cross-Val Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importances
    feat_imp = dict(zip(feature_cols, rf_model.feature_importances_))
    feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    # Save models and encoders
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "random_forest.pkl"))
    joblib.dump(dt_model, os.path.join(MODEL_DIR, "decision_tree.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))

    metadata = {
        "feature_cols": feature_cols,
        "rf_accuracy": rf_acc,
        "dt_accuracy": dt_acc,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "feature_importances": {k: float(v) for k, v in feat_imp_sorted.items()},
        "all_subjects": ALL_SUBJECTS,
        "all_skills": ALL_SKILLS,
        "career_options": CAREER_OPTIONS,
        "interest_options": INTEREST_OPTIONS,
        "skill_levels": SKILL_LEVELS
    }

    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nModels saved successfully!")
    print(f"Saved to: {MODEL_DIR}")
    return rf_model, dt_model, encoders, feature_cols


if __name__ == "__main__":
    train()
