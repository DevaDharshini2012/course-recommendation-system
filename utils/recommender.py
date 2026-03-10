"""
Course Recommendation Engine with Explainable AI (XAI)
Uses Random Forest (primary) + Decision Tree (explainability)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


class CourseRecommender:
    def __init__(self):
        self.rf_model = None
        self.dt_model = None
        self.encoders = None
        self.metadata = None
        self.courses_df = None
        self._loaded = False

    def load(self):
        """Load models, encoders, and course data."""
        if self._loaded:
            return

        rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
        if not os.path.exists(rf_path):
            raise FileNotFoundError(
                "Model not found. Please run 'python train_model.py' first."
            )

        self.rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
        self.dt_model = joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl"))
        self.encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))

        with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

        self.courses_df = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
        self._loaded = True

    def recommend(self, user_data: dict) -> dict:
        """
        Generate course recommendation with explanation.

        user_data keys:
        - subject_marks: {subject: mark}
        - logical_thinking: float
        - analytical_ability: float
        - programming_score: float
        - skill_level: str  (beginner/intermediate/advanced)
        - interest: str
        - career_goal: str
        - user_id: str (optional)
        """
        self.load()

        feature_vector = self._build_feature_vector(user_data)
        feature_df = pd.DataFrame([feature_vector], columns=self.metadata["feature_cols"])

        # Primary prediction: Random Forest
        rf_proba = self.rf_model.predict_proba(feature_df)[0]
        rf_pred = self.rf_model.predict(feature_df)[0]

        # Decode prediction
        le_target = self.encoders["target"]
        predicted_course_id = le_target.inverse_transform([rf_pred])[0]

        # Top-3 alternatives
        top3_indices = np.argsort(rf_proba)[::-1][:3]
        top3_courses = []
        for idx in top3_indices:
            course_id = le_target.inverse_transform([idx])[0]
            course_info = self._get_course_info(course_id)
            if course_info:
                top3_courses.append({
                    "course_id": course_id,
                    "course_name": course_info["course_name"],
                    "confidence": round(float(rf_proba[idx]) * 100, 1),
                    "category": course_info["category"],
                    "level": course_info["level"]
                })

        # Get primary course details
        primary_course = self._get_course_info(predicted_course_id)

        # Generate explanation
        explanation = self._generate_explanation(
            user_data, feature_vector, feature_df, primary_course
        )

        result = {
            "recommended_course": primary_course,
            "confidence": round(float(rf_proba[rf_pred]) * 100, 1),
            "alternative_courses": top3_courses[1:],  # exclude primary
            "explanation": explanation,
            "user_profile_summary": {
                "skill_level": user_data.get("skill_level"),
                "interest": user_data.get("interest"),
                "career_goal": user_data.get("career_goal"),
                "overall_average": user_data.get("overall_average")
            },
            "timestamp": datetime.now().isoformat()
        }

        return result

    def _build_feature_vector(self, user_data: dict) -> list:
        """Build numeric feature vector from user data."""
        all_subjects = self.metadata["all_subjects"]
        subject_marks = user_data.get("subject_marks", {})

        features = []

        # Subject marks
        for subj in all_subjects:
            features.append(float(subject_marks.get(subj, 65.0)))

        # Derived scores
        features.append(float(user_data.get("overall_average", 65.0)))
        features.append(float(user_data.get("logical_thinking", 65.0)))
        features.append(float(user_data.get("analytical_ability", 65.0)))
        features.append(float(user_data.get("programming_score", 65.0)))

        # Categorical encodings
        skill_level = user_data.get("skill_level", "beginner")
        interest = user_data.get("interest", "data_science")
        career = user_data.get("career_goal", "Data Scientist")

        le_skill = self.encoders["skill_level"]
        le_interest = self.encoders["interest"]
        le_career = self.encoders["career_goal"]

        # Safe encoding with fallback (ensuring python int for JSON serialization)
        try:
            skill_enc = int(le_skill.transform([skill_level])[0])
        except ValueError:
            skill_enc = 0

        try:
            interest_enc = int(le_interest.transform([interest])[0])
        except ValueError:
            interest_enc = 0

        try:
            career_enc = int(le_career.transform([career])[0])
        except ValueError:
            career_enc = 0

        features.extend([skill_enc, interest_enc, career_enc])
        return features

    def _get_course_info(self, course_id: str) -> dict:
        """Retrieve course details from dataset and ensure JSON serializable types."""
        row = self.courses_df[self.courses_df["course_id"] == course_id]
        if row.empty:
            return None
        
        # Convert Series to dict and cast numpy types to python types
        course_dict = row.iloc[0].to_dict()
        clean_dict = {}
        for k, v in course_dict.items():
            if hasattr(v, "item"):  # Generic way to handles numpy scalars
                clean_dict[k] = v.item()
            elif pd.isna(v):
                clean_dict[k] = None
            else:
                clean_dict[k] = v
        return clean_dict

    def _generate_explanation(self, user_data: dict, feature_vector: list,
                               feature_df: pd.DataFrame, course: dict) -> dict:
        """
        Generate XAI explanation using:
        1. Feature importances from Random Forest
        2. Decision path from Decision Tree
        3. Rule-based natural language explanation
        """
        feature_cols = self.metadata["feature_cols"]
        feature_importances = self.metadata["feature_importances"]

        # Top contributing features
        feat_vals = dict(zip(feature_cols, feature_vector))
        top_features = []
        for feat, importance in list(feature_importances.items())[:8]:
            val = feat_vals.get(feat, 0)
            top_features.append({
                "feature": feat,
                "importance": round(importance * 100, 2),
                "your_value": round(val, 1),
                "interpretation": self._interpret_feature(feat, val)
            })

        # Natural language reason
        reasons = self._build_natural_language_reasons(user_data, course)

        # Decision tree path
        dt_path = self._get_dt_decision_path(feature_df)

        return {
            "natural_language_reasons": reasons,
            "top_contributing_features": top_features,
            "decision_tree_path": dt_path,
            "model_used": "Random Forest (200 trees) with Decision Tree for interpretability",
            "recommendation_factors": {
                "interest_alignment": user_data.get("interest", ""),
                "career_alignment": user_data.get("career_goal", ""),
                "skill_level_match": user_data.get("skill_level", ""),
                "academic_strength": round(user_data.get("overall_average", 65), 1)
            }
        }

    def _interpret_feature(self, feature: str, value: float) -> str:
        """Human-readable interpretation of a feature value."""
        if "skill_level" in feature:
            levels = {0: "beginner", 1: "intermediate", 2: "advanced"}
            return f"Skill level: {levels.get(int(value), 'beginner')}"
        if "interest" in feature:
            return "Your primary area of interest"
        if "career_goal" in feature:
            return "Your target career path"
        if value >= 80:
            return "Excellent performance"
        elif value >= 65:
            return "Good performance"
        elif value >= 50:
            return "Average performance"
        else:
            return "Needs improvement"

    def _build_natural_language_reasons(self, user_data: dict, course: dict) -> list:
        """Build natural language explanation for the recommendation."""
        reasons = []
        skill = user_data.get("skill_level", "beginner")
        interest = user_data.get("interest", "").replace("_", " ")
        career = user_data.get("career_goal", "")
        avg = user_data.get("overall_average", 65)
        logical = user_data.get("logical_thinking", 65)
        analytical = user_data.get("analytical_ability", 65)
        prog = user_data.get("programming_score", 65)

        if course:
            course_level = course.get("level", "beginner")
            course_name = course.get("course_name", "")
            course_category = course.get("category", "")
            career_path = course.get("career_path", "")

            reasons.append(
                f"Your interest in {interest} strongly aligns with the {course_category} category "
                f"of '{course_name}'."
            )

            if career and career in career_path:
                reasons.append(
                    f"This course directly prepares you for your career goal as a {career}, "
                    f"which is listed as a career path for this course."
                )

            if skill == course_level:
                reasons.append(
                    f"Your current programming skill level ({skill}) perfectly matches "
                    f"the required level ({course_level}) for this course."
                )
            elif skill == "beginner" and course_level == "beginner":
                reasons.append(
                    "As a beginner, this foundational course will build your skills progressively."
                )

            if logical >= 70:
                reasons.append(
                    f"Your logical thinking score ({logical:.0f}/100) indicates strong problem-solving "
                    f"ability suited for this course."
                )
            if analytical >= 70:
                reasons.append(
                    f"Your analytical ability ({analytical:.0f}/100) is a key strength "
                    f"for success in this domain."
                )
            if prog >= 70:
                reasons.append(
                    f"Your programming fundamentals score ({prog:.0f}/100) shows you have the "
                    f"technical foundation needed."
                )
            if avg >= 75:
                reasons.append(
                    f"Your overall academic performance ({avg:.0f}/100) demonstrates the "
                    f"discipline required to excel in this course."
                )

        return reasons

    def _get_dt_decision_path(self, feature_df: pd.DataFrame) -> list:
        """Extract simplified decision path from the Decision Tree."""
        from sklearn.tree import _tree
        feature_cols = self.metadata["feature_cols"]
        tree = self.dt_model.tree_

        path = []
        node = 0
        max_depth = 6

        for _ in range(max_depth):
            if tree.feature[node] == _tree.TREE_UNDEFINED:
                break

            feat_idx = tree.feature[node]
            feat_name = feature_cols[feat_idx] if feat_idx < len(feature_cols) else f"feature_{feat_idx}"
            threshold = tree.threshold[node]
            feat_val = float(feature_df.iloc[0, feat_idx]) if feat_idx < len(feature_df.columns) else 0

            direction = "≤" if feat_val <= threshold else ">"
            went_left = feat_val <= threshold

            path.append({
                "feature": feat_name,
                "threshold": round(threshold, 2),
                "your_value": round(feat_val, 2),
                "direction": direction,
                "description": f"{feat_name} ({feat_val:.1f}) {direction} {threshold:.1f}"
            })

            node = tree.children_left[node] if went_left else tree.children_right[node]

        return path


# Global instance
_recommender = CourseRecommender()


def get_recommendation(user_data: dict) -> dict:
    """Public API function for generating recommendations."""
    return _recommender.recommend(user_data)


def get_model_info() -> dict:
    """Return model metadata for admin dashboard."""
    _recommender.load()
    return _recommender.metadata
