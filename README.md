# ML-Based Course Recommendation System with Explainable AI

## Project Overview
A machine learning-based web application that recommends personalized courses to students based on:
1. Academic performance (extracted from PDF marksheets or manually entered)
2. Programming skill assessment (5-question coding test)
3. Personal interests and career goals

The system uses a **Random Forest** model for recommendations and a **Decision Tree** for interpretability (XAI), providing full transparency on why each course is recommended.

---

## Project Structure

```
course_recommender/
├── app.py                    # Main Flask application
├── train_model.py            # ML model training script (run first!)
├── requirements.txt          # Python dependencies
├── README.md
│
├── data/
│   ├── courses.csv           # Static course dataset (25 courses)
│   ├── users.csv             # Generated: user profile records (admin only)
│   └── recommendations.csv   # Generated: recommendation results (admin only)
│
├── models/                   # Generated after training
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── encoders.pkl
│   └── metadata.json
│
├── utils/
│   ├── pdf_parser.py         # PDF marksheet extraction
│   ├── skill_test.py         # Coding test question bank & evaluator
│   ├── recommender.py        # ML engine + XAI explanation generator
│   └── data_manager.py       # CSV read/write manager
│
├── templates/
│   ├── base.html
│   ├── index.html            # Landing page
│   ├── start.html            # Step 1: Basic info
│   ├── upload.html           # Step 2: Marksheet upload/manual entry
│   ├── skill_test.html       # Step 3: Coding test
│   ├── interests.html        # Step 4: Interests & career goals
│   ├── recommendation.html   # Step 5: Results with XAI
│   └── admin/
│       ├── login.html
│       ├── dashboard.html
│       ├── users.html
│       ├── recommendations.html
│       └── model_info.html
│
└── uploads/                  # Temporary PDF storage (auto-cleaned)
```

---

## Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the ML Model (REQUIRED before running the app)
```bash
python train_model.py
```
This generates synthetic training data (3000 samples), trains both models, and saves them to `models/`.

### 3. Run the Application
```bash
python app.py
```
Open: `http://localhost:5000`

---

## User Flow

```
Home → Start (Name/Email) → Upload Marksheets or Enter Marks
     → Programming Skill Test (5 questions, 30 min)
     → Interest & Career Goals
     → AI Recommendation + Full XAI Explanation
```

---

## Admin Access

- URL: `http://localhost:5000/admin/login`
- Default credentials: `admin / admin123`
- **Change these in `app.py` before deployment!**

Admin features:
- View all user profiles
- View all recommendation results with XAI reasoning
- Download users.csv and recommendations.csv
- View model performance metrics and feature importances

---

## ML Architecture

### Models Used
| Model | Role | Accuracy |
|-------|------|----------|
| Random Forest (200 trees) | Primary recommendation | ~95%+ |
| Decision Tree (depth 10) | Interpretability / XAI | ~90%+ |

### Features (19 total)
- **Subject marks**: Mathematics, Physics, Chemistry, CS, English, Statistics, Biology, Economics
- **Derived scores**: Overall average, Logical thinking, Analytical ability, Programming score
- **Categorical**: Skill level (beginner/intermediate/advanced), Interest area, Career goal

### XAI Explanation Components
1. **Natural Language Reasons**: Human-readable bullet points explaining the recommendation
2. **Feature Importances**: Which features mattered most (from Random Forest)
3. **Decision Tree Path**: Step-by-step branching logic from the interpretable DT model
4. **Alternative Courses**: Top-3 course suggestions with confidence scores

---

## Course Dataset
25 static courses across categories:
- Data Science, AI/ML, Web Development, Mobile Development
- Cloud Computing, DevOps, Cybersecurity, Database
- Blockchain, Game Development, UI/UX Design, Management

---

## Key Design Decisions

- **Static course dataset**: Stored in `courses.csv`, not modified by user interactions
- **Dynamic user data**: Stored in append-only CSV files, admin-access only
- **PDF extraction**: Uses `pdfplumber` with PyPDF2 fallback; falls back gracefully
- **No external API**: Fully offline-capable after setup
- **Session-based flow**: Flask sessions track multi-step progress
- **Feature engineering**: Derived strength scores from raw marks (logical, analytical, programming)

---

## Customization

### Adding New Courses
Edit `data/courses.csv` and retrain: `python train_model.py`

### Adding New Questions
Edit `utils/skill_test.py` → `QUESTION_BANK` dictionary.

### Changing Admin Credentials
Edit `ADMIN_USERNAME` and `ADMIN_PASSWORD` in `app.py`.

### Retraining with Real Data
Replace synthetic data generation in `train_model.py` with your real dataset.
