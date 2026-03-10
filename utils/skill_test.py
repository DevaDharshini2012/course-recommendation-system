"""
Programming Skill Test Evaluator
Evaluates user responses to coding questions and determines skill level.
"""

import json
import random


# ─────────────────────────────────────────────────────────────
# Question Bank
# ─────────────────────────────────────────────────────────────
QUESTION_BANK = {
    "python": {
        "beginner": [
            {
                "id": "py_b1",
                "question": "Write a Python function to check if a number is even or odd.",
                "expected_concepts": ["def", "if", "else", "return", "% 2"],
                "hint": "Use the modulo operator %",
                "difficulty": "beginner"
            },
            {
                "id": "py_b2",
                "question": "Write a Python program to find the factorial of a number using a loop.",
                "expected_concepts": ["for", "while", "range", "factorial", "*="],
                "hint": "Use a loop to multiply numbers from 1 to n",
                "difficulty": "beginner"
            },
            {
                "id": "py_b3",
                "question": "Write a Python function that takes a list and returns the sum of all elements.",
                "expected_concepts": ["def", "sum", "for", "append", "return"],
                "hint": "You can use built-in sum() or iterate through the list",
                "difficulty": "beginner"
            }
        ],
        "intermediate": [
            {
                "id": "py_i1",
                "question": "Implement a binary search algorithm in Python.",
                "expected_concepts": ["def", "while", "mid", "left", "right", "return", "//"],
                "hint": "Divide the search space in half each iteration",
                "difficulty": "intermediate"
            },
            {
                "id": "py_i2",
                "question": "Write a Python class for a Stack data structure with push, pop, and peek operations.",
                "expected_concepts": ["class", "def", "__init__", "self", "append", "pop", "[-1]"],
                "hint": "Use a list as the underlying data structure",
                "difficulty": "intermediate"
            }
        ],
        "advanced": [
            {
                "id": "py_a1",
                "question": "Implement a function to find the longest common subsequence of two strings using dynamic programming.",
                "expected_concepts": ["dp", "matrix", "for", "if", "max", "len", "return"],
                "hint": "Use a 2D table where dp[i][j] stores LCS length",
                "difficulty": "advanced"
            },
            {
                "id": "py_a2",
                "question": "Write a Python decorator that measures the execution time of a function.",
                "expected_concepts": ["def", "wrapper", "functools", "wraps", "time", "return"],
                "hint": "Use time.time() before and after the function call",
                "difficulty": "advanced"
            }
        ]
    },
    "java": {
        "beginner": [
            {
                "id": "ja_b1",
                "question": "Write a Java method to reverse a string without using built-in reverse functions.",
                "expected_concepts": ["for", "charAt", "StringBuilder", "append", "return"],
                "hint": "Iterate from the end of the string to the beginning",
                "difficulty": "beginner"
            },
            {
                "id": "ja_b2",
                "question": "Write a Java program to check if a number is prime.",
                "expected_concepts": ["for", "if", "modulo", "%", "return", "boolean"],
                "hint": "Check divisibility up to square root of the number",
                "difficulty": "beginner"
            },
            {
                "id": "ja_b3",
                "question": "Write a Java method to find the maximum element in an array.",
                "expected_concepts": ["for", "if", "int", "max", "return", "array"],
                "hint": "Initialize max with first element, then compare with rest",
                "difficulty": "beginner"
            }
        ],
        "intermediate": [
            {
                "id": "ja_i1",
                "question": "Implement a LinkedList with add and delete operations in Java.",
                "expected_concepts": ["class", "Node", "next", "head", "while", "null"],
                "hint": "Create a Node class with data and next pointer",
                "difficulty": "intermediate"
            },
            {
                "id": "ja_i2",
                "question": "Write a Java program to implement bubble sort algorithm.",
                "expected_concepts": ["for", "swap", "if", "temp", "array", "nested"],
                "hint": "Compare adjacent elements and swap if needed",
                "difficulty": "intermediate"
            }
        ],
        "advanced": [
            {
                "id": "ja_a1",
                "question": "Implement a generic stack using Java generics.",
                "expected_concepts": ["class", "<T>", "Stack", "push", "pop", "ArrayList", "Exception"],
                "hint": "Use ArrayList<T> as the backing store with type parameter T",
                "difficulty": "advanced"
            },
            {
                "id": "ja_a2",
                "question": "Write a Java program to implement Dijkstra's shortest path algorithm.",
                "expected_concepts": ["PriorityQueue", "HashMap", "graph", "distance", "visited", "for"],
                "hint": "Use a priority queue to always process the shortest distance node",
                "difficulty": "advanced"
            }
        ]
    },
    "c": {
        "beginner": [
            {
                "id": "c_b1",
                "question": "Write a C function to swap two integers using pointers.",
                "expected_concepts": ["*", "&", "temp", "void", "int"],
                "hint": "Use pointers and a temporary variable",
                "difficulty": "beginner"
            },
            {
                "id": "c_b2",
                "question": "Write a C program to find the GCD of two numbers.",
                "expected_concepts": ["while", "if", "%", "return", "int"],
                "hint": "Use the Euclidean algorithm: gcd(a,b) = gcd(b, a%b)",
                "difficulty": "beginner"
            },
            {
                "id": "c_b3",
                "question": "Write a C function to count the number of vowels in a string.",
                "expected_concepts": ["for", "if", "char", "count", "return"],
                "hint": "Check each character against a, e, i, o, u",
                "difficulty": "beginner"
            }
        ],
        "intermediate": [
            {
                "id": "c_i1",
                "question": "Implement a stack using arrays in C with push and pop functions.",
                "expected_concepts": ["struct", "int", "top", "push", "pop", "overflow", "underflow"],
                "hint": "Use a struct with an array and top index",
                "difficulty": "intermediate"
            },
            {
                "id": "c_i2",
                "question": "Write a C program to implement matrix multiplication.",
                "expected_concepts": ["for", "int", "matrix", "result", "nested", "loop"],
                "hint": "Use three nested loops: row, column, and inner product",
                "difficulty": "intermediate"
            }
        ],
        "advanced": [
            {
                "id": "c_a1",
                "question": "Write a C program to implement a binary tree with insert and in-order traversal.",
                "expected_concepts": ["struct", "Node", "*left", "*right", "malloc", "recursive", "inorder"],
                "hint": "Use recursive functions for tree operations",
                "difficulty": "advanced"
            }
        ]
    }
}


def get_test_questions(language: str, n: int = 5) -> list:
    """
    Select n questions for the skill test from the given language.
    Mix of difficulties: 2 beginner, 2 intermediate, 1 advanced.
    """
    language = language.lower()
    if language not in QUESTION_BANK:
        language = "python"  # default

    bank = QUESTION_BANK[language]
    questions = []

    # Pull from each difficulty level
    for level in ["beginner", "intermediate", "advanced"]:
        qs = bank.get(level, [])
        questions.extend(qs)

    # Shuffle and return n questions
    random.shuffle(questions)
    return questions[:n]


def evaluate_submission(questions: list, answers: dict) -> dict:
    """
    Evaluate user answers to the coding test.

    answers: {question_id: answer_text}
    Returns: {score, skill_level, per_question_feedback, total_questions}
    """
    total = len(questions)
    score = 0
    feedback = []

    for q in questions:
        qid = q["id"]
        answer = answers.get(qid, "").lower().strip()
        expected = q["expected_concepts"]
        difficulty = q["difficulty"]

        # Score based on concept keywords found in answer
        concepts_found = sum(1 for concept in expected if concept.lower() in answer)
        concept_ratio = concepts_found / len(expected) if expected else 0

        # Weight by difficulty
        weights = {"beginner": 1.0, "intermediate": 1.5, "advanced": 2.0}
        weight = weights.get(difficulty, 1.0)

        # Minimum answer length check
        if len(answer.split()) < 3:
            q_score = 0
        else:
            q_score = concept_ratio * weight

        score += q_score

        per_q_result = {
            "question_id": qid,
            "question": q["question"],
            "difficulty": difficulty,
            "concepts_matched": concepts_found,
            "total_concepts": len(expected),
            "score": round(q_score, 2),
            "feedback": _generate_feedback(concept_ratio, difficulty, answer)
        }
        feedback.append(per_q_result)

    # Normalize score to 0-100
    max_possible = sum(
        {"beginner": 1.0, "intermediate": 1.5, "advanced": 2.0}.get(q["difficulty"], 1.0)
        for q in questions
    )
    normalized_score = (score / max_possible * 100) if max_possible > 0 else 0
    normalized_score = round(normalized_score, 2)

    # Determine skill level
    if normalized_score < 35:
        skill_level = "beginner"
    elif normalized_score < 65:
        skill_level = "intermediate"
    else:
        skill_level = "advanced"

    return {
        "total_questions": total,
        "raw_score": round(score, 2),
        "normalized_score": normalized_score,
        "skill_level": skill_level,
        "per_question_feedback": feedback
    }


def _generate_feedback(concept_ratio: float, difficulty: str, answer: str) -> str:
    """Generate human-readable feedback for a question."""
    if not answer or len(answer.split()) < 3:
        return "No answer provided or answer too short."
    if concept_ratio >= 0.8:
        return "Excellent! You demonstrated strong understanding of the key concepts."
    elif concept_ratio >= 0.5:
        return "Good attempt. You covered most key concepts but could improve further."
    elif concept_ratio >= 0.2:
        return "Partial solution. Review the core concepts for this topic."
    else:
        return "Needs improvement. Study the fundamental concepts and try again."
