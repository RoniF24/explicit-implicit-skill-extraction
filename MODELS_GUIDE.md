# 📘 SkillSight - מדריך הפעלת מודלים

## 🎯 סקירה כללית

הפרויקט כולל **3 מודלים** לזיהוי מיומנויות מטקסט:

| מודל | דיוק (F1) | מהירות | המלצה |
|------|-----------|---------|--------|
| **DeBERTa Pairwise** | 97% | איטי | ✅ **מומלץ לשימוש** |
| **RoBERTa Pairwise** | 95% | איטי | טוב |
| **MODELV2 One-Pass** | 48% | מהיר | ❌ לא מומלץ |

---

## 📁 מבנה הקבצים

```
SkillSight/
├── models/
│   ├── roberta_base/          # מודל RoBERTa מאומן
│   └── deberta_v3_base/       # מודל DeBERTa מאומן (הכי טוב!)
├── MODELV2/
│   └── experiments/           # מודל MODELV2 (לא מומלץ)
├── src/
│   └── skills/
│       └── skills_v1.txt      # רשימת 136 מיומנויות אפשריות
├── results_of_model/          # תוצאות ההרצות נשמרות כאן
├── analyze_resume.py          # סקריפט לניתוח טקסט בודד
├── batch_analyze.py           # סקריפט לניתוח קובץ עם מספר טקסטים
└── compare_models_demo.py     # השוואה בין כל המודלים
```

---

## 🚀 התקנה והפעלה ראשונית

```powershell
# 1. כניסה לתיקיית הפרויקט
cd C:\NLP\SkillSight

# 2. הפעלת הסביבה הוירטואלית
.\.venv\Scripts\Activate.ps1

# 3. בדיקה שהכל עובד
python analyze_resume.py --help
```

---

# 📝 סקריפט 1: analyze_resume.py

## ניתוח טקסט בודד

### שימוש בסיסי

```powershell
python analyze_resume.py --text "הטקסט שלך כאן" --model deberta
```

### פרמטרים

| פרמטר | תיאור | ברירת מחדל |
|-------|--------|-------------|
| `--text` | הטקסט לניתוח (חובה) | - |
| `--model` | איזה מודל: `deberta`, `roberta`, `all` | `deberta` |
| `--ground-truth` | קובץ JSON או מחרוזת עם תוצאות צפויות | - |

### דוגמאות

#### 1️⃣ ניתוח פשוט עם DeBERTa (הכי טוב)

```powershell
python analyze_resume.py --text "I have 5 years of experience with Python and Django. Built REST APIs and deployed using Docker and Kubernetes on AWS." --model deberta
```

**פלט צפוי:**
```
DETECTED SKILLS:

  EXPLICIT (1.0) - 5 skills:
    • AWS
    • Django
    • Docker
    • Kubernetes
    • Python

  IMPLICIT (0.5) - 1 skills:
    • REST API Design
```

#### 2️⃣ ניתוח עם RoBERTa

```powershell
python analyze_resume.py --text "Experienced in Java and Spring Boot, building microservices with PostgreSQL databases." --model roberta
```

#### 3️⃣ השוואת כל המודלים

```powershell
python analyze_resume.py --text "Full stack developer with React, Node.js and MongoDB experience." --model all
```

#### 4️⃣ ניתוח עם Ground Truth (בדיקת דיוק)

יצירת קובץ ground truth:
```powershell
# יצירת קובץ JSON עם התוצאות הצפויות
echo '{"Python": 1.0, "Django": 1.0, "Docker": 0.5}' > expected.json
```

הרצה עם בדיקת דיוק:
```powershell
python analyze_resume.py --text "I work with Python and Django daily. I also containerize my applications." --model deberta --ground-truth expected.json
```

**פלט עם מדדי דיוק:**
```
ACCURACY ANALYSIS vs GROUND TRUTH:
----------------------------------------
  Precision: 85.0%
  Recall:    100.0%
  F1 Score:  91.9%
  
  True Positives (3): Django, Docker, Python
  False Positives (1): PostgreSQL
  False Negatives (0): 
```

---

# 📊 סקריפט 2: batch_analyze.py

## ניתוח קובץ עם מספר טקסטים

### פורמט קובץ הקלט (JSONL)

כל שורה היא JSON עם השדות:
- `job_description` או `text` - הטקסט לניתוח
- `skills` או `ground_truth` - (אופציונלי) מיומנויות צפויות

**דוגמה - `my_texts.jsonl`:**
```json
{"text": "Python developer with Flask experience", "ground_truth": {"Python": 1.0, "Flask": 0.5}}
{"text": "DevOps engineer using Docker and Kubernetes", "ground_truth": {"Docker": 1.0, "Kubernetes": 1.0}}
{"text": "Data scientist with SQL and Apache Spark skills", "ground_truth": {"SQL": 1.0, "Apache Spark": 1.0}}
```

### שימוש

```powershell
python batch_analyze.py --input my_texts.jsonl --output batch_results.txt
```

### פרמטרים

| פרמטר | תיאור | ברירת מחדל |
|-------|--------|-------------|
| `--input` | קובץ JSONL לניתוח (חובה) | - |
| `--output` | שם קובץ הפלט | `batch_results.txt` |
| `--limit` | מספר מקסימלי של דוגמאות | ללא הגבלה |
| `--model` | `roberta`, `deberta`, `all` | `all` |

### דוגמאות

#### 1️⃣ ניתוח מהדאטאסט שלך

```powershell
python batch_analyze.py --input data/splits_v1/test.jsonl --limit 10
```

#### 2️⃣ ניתוח קובץ מותאם אישית

```powershell
python batch_analyze.py --input my_resumes.jsonl --model deberta --output my_analysis.txt
```

#### 3️⃣ ניתוח מהיר (5 דוגמאות בלבד)

```powershell
python batch_analyze.py --input data/synthetic_dataset.jsonl --limit 5 --model deberta
```

---

# 🔄 סקריפט 3: compare_models_demo.py

## השוואה מלאה בין כל המודלים

### שימוש

```powershell
python compare_models_demo.py --text "Your text here" --output comparison.txt
```

### פרמטרים

| פרמטר | תיאור | ברירת מחדל |
|-------|--------|-------------|
| `--text` | טקסט לניתוח | דוגמה מובנית |
| `--file` | קרא טקסט מקובץ | - |
| `--output` | שם קובץ הפלט | `comparison_results.txt` |
| `--ground-truth` | קובץ JSON עם תוצאות צפויות | - |

### דוגמאות

#### 1️⃣ השוואה בסיסית

```powershell
python compare_models_demo.py --text "Senior engineer with Python, AWS, and Docker expertise."
```

#### 2️⃣ השוואה מקובץ טקסט

```powershell
# יצירת קובץ טקסט
echo "I am a backend developer specializing in Python and PostgreSQL. Experience with Docker containers and CI/CD pipelines." > resume.txt

# הרצה
python compare_models_demo.py --file resume.txt --output my_comparison.txt
```

---

# 📂 מיקום התוצאות

כל התוצאות נשמרות בתיקייה:
```
results_of_model/
├── analysis_deberta_20260103_210615.txt
├── analysis_roberta_20260103_211234.txt
└── analysis_all_models_20260103_212345.txt
```

שם הקובץ כולל:
- סוג המודל (`deberta`, `roberta`, `all_models`)
- תאריך ושעה

---

# 🛠️ יצירת קובץ קלט משלך

## פורמט 1: JSONL פשוט (ללא ground truth)

```json
{"text": "Python developer with 3 years experience"}
{"text": "DevOps engineer familiar with Docker and Kubernetes"}
{"text": "Data analyst using SQL and Tableau"}
```

## פורמט 2: JSONL עם ground truth

```json
{"text": "Python developer", "ground_truth": {"Python": 1.0}}
{"text": "Uses Docker daily", "ground_truth": {"Docker": 1.0, "Kubernetes": 0.5}}
```

## פורמט 3: כמו הדאטאסט המקורי

```json
{"job_description": "...", "skills": {"Python": 1.0, "Flask": 0.5}}
```

---

# 📋 רשימת המיומנויות הנתמכות

המודלים יכולים לזהות **136 מיומנויות** בלבד!

לצפייה ברשימה המלאה:
```powershell
type src\skills\skills_v1.txt
```

### קטגוריות עיקריות:

| קטגוריה | דוגמאות |
|---------|---------|
| **שפות תכנות** | Python, Java, JavaScript, TypeScript, Go, Rust |
| **Frameworks** | Django, Flask, FastAPI, Spring Boot, React, Node.js |
| **Databases** | PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch |
| **DevOps** | Docker, Kubernetes, Terraform, Jenkins, GitHub Actions |
| **Cloud** | AWS, Azure, Google Cloud |
| **Security** | OWASP Top 10, Penetration Testing, SIEM, Network Security |
| **Testing** | Unit Testing, API Testing, Selenium, Playwright |
| **Data** | Apache Spark, Apache Kafka, SQL, dbt |

---

# ⚠️ דברים חשובים לזכור

1. **מיומנויות חייבות להיות מהרשימה** - המודל לא יזהה מיומנויות שלא ב-`skills_v1.txt`

2. **סיווג הציונים:**
   - `1.0` = EXPLICIT - המיומנות מוזכרת ישירות בטקסט
   - `0.5` = IMPLICIT - המיומנות משתמעת מההקשר
   - `0.0` = NONE - המיומנות לא קיימת

3. **DeBERTa הכי טוב** - השתמש תמיד ב-`--model deberta` לתוצאות הכי טובות

4. **False Positives** - המודל עלול לזהות מיומנויות שלא קיימות (precision לא מושלם)

5. **תוצאות נשמרות אוטומטית** ב-`results_of_model/`

---

# 🎯 דוגמה מלאה - מקרה שימוש אמיתי

```powershell
# 1. הכנת קובץ עם טקסטים לניתוח
@"
{"text": "Senior Python developer with Django and PostgreSQL. Built REST APIs and deployed on AWS using Docker.", "ground_truth": {"Python": 1.0, "Django": 1.0, "PostgreSQL": 1.0, "REST API Design": 1.0, "AWS": 1.0, "Docker": 1.0}}
{"text": "DevOps engineer experienced with Kubernetes, Terraform, and CI/CD pipelines using GitHub Actions.", "ground_truth": {"Kubernetes": 1.0, "Terraform": 1.0, "GitHub Actions": 1.0}}
{"text": "Security analyst focusing on penetration testing and vulnerability scanning using Burp Suite.", "ground_truth": {"Penetration Testing": 1.0, "Vulnerability Scanning": 1.0, "Burp Suite": 0.5}}
"@ | Out-File -Encoding UTF8 my_test.jsonl

# 2. הרצת ניתוח
python batch_analyze.py --input my_test.jsonl --model deberta

# 3. צפייה בתוצאות
type batch_results.txt
```

---

# 📞 עזרה

```powershell
# עזרה לכל סקריפט
python analyze_resume.py --help
python batch_analyze.py --help
python compare_models_demo.py --help
```

---

**נוצר ע"י SkillSight Team | ינואר 2026**
