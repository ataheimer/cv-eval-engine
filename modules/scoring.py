import math

def score_skills(n):
    return min(50, round(50 * math.log1p(n) / math.log1p(10)))

def score_experience(years):
    return min(30, years * 3)

def score_education(level):
    return {"PhD":20, "MSc":16, "BSc":12}.get(level, 6)

def total_score(skills_found, years, edu):
    return score_skills(len(skills_found)) + score_experience(years) + score_education(edu)
