
# Building an AI Talent Demand Tracker: A Software Developer's Workflow for HR Insights

## Introduction to the Case Study

Welcome to your next lab for building the "PE-Org AIR System"! Your mission is to empower the HR and strategy teams with crucial insights into the evolving demand for AI talent in the broader market. The goal is to build an internal tool that provides actionable intelligence, enabling proactive adaptation of hiring strategies, focus on in-demand skills, and a clear understanding of the competitive landscape for AI professionals. This tool is vital for the company to stay agile and competitive in a fast-changing talent market.

Today, you'll be developing the core components of this system. This involves simulating the data collection, implementing sophisticated logic for extracting AI-related skills and classifying job seniority, calculating a critical AI relevance score, and finally, aggregating this data for time-series analysis and visualization. Every step you take is designed to directly address a business need, transforming raw data into strategic insights for your organization.

### Learning Objectives

By the end of this notebook, you will have implemented a system that:
*   Collects (simulated) job posting data, handling real-world issues like deduplication.
*   Applies a comprehensive AI skill taxonomy to extract relevant skills from job descriptions.
*   Classifies job seniority levels based on job titles.
*   Calculates an AI relevance score for each job posting using a predefined formula.
*   Aggregates data weekly to enable time-series analysis of AI talent market trends.
*   Visualizes key insights, such as weekly job volume, top skills, and seniority distribution.

---

## 1. Environment Setup and Data Generation

As a Software Developer, the first step is always to ensure your environment is ready and that you have data to work with. Since live scraping can be complex and rate-limited within a single notebook, we'll simulate the data acquisition phase by generating a diverse set of synthetic job postings that mirror real-world data. This allows us to focus on the processing and analysis logic, which is your primary task.

### 1.1 Install Required Libraries

Before writing any code, we need to install the necessary Python libraries. These include `pandas` for data manipulation, `Faker` for generating realistic synthetic data, `matplotlib` and `seaborn` for visualizations, and `numpy` for numerical operations.

```python
!pip install pandas faker matplotlib seaborn numpy
```

### 1.2 Import Dependencies

Next, import all the Python libraries and modules that will be used throughout the notebook. This makes sure all necessary functionalities are available from the start.

```python
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from typing import List, Set, Dict, Tuple
from enum import Enum

# Configure plot styles for better readability
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')
```

### 1.3 Simulate Job Posting Data Collection

The HR team needs a robust dataset to analyze. Your task is to generate at least 1,500 unique job postings, each with a title, a full description, the source, a URL, and a realistic `posted_date`. This synthetic dataset will serve as the foundation for all subsequent analysis, simulating the output of a job scraper.

```python
def generate_synthetic_job_postings(num_postings: int = 1500) -> List[Dict]:
    """
    Generates a list of synthetic job postings, simulating scraped data.

    Args:
        num_postings (int): The number of job postings to generate.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a job posting.
    """
    fake = Faker('en_US')
    job_postings = []

    # Define common AI-related job titles and descriptions
    ai_titles = [
        "Machine Learning Engineer", "Data Scientist", "AI Research Scientist",
        "Deep Learning Engineer", "NLP Engineer", "Computer Vision Engineer",
        "MLOps Engineer", "AI Product Manager", "AI/ML Software Engineer",
        "Applied Scientist (AI)", "Senior Data Scientist", "Lead ML Engineer",
        "AI Infrastructure Engineer", "Quantitative Researcher (ML)",
        "AI Strategist", "Robotics Engineer (AI)", "Generative AI Engineer",
        "AI Solutions Architect", "Principal Machine Learning Engineer"
    ]
    non_ai_titles = [
        "Software Engineer", "Frontend Developer", "Backend Developer",
        "DevOps Engineer", "Product Manager", "UX Designer",
        "Project Manager", "Data Analyst", "Business Analyst"
    ]

    ai_keywords = [
        "pytorch", "tensorflow", "keras", "mlops", "machine learning", "deep learning",
        "nlp", "computer vision", "transformers", "reinforcement learning",
        "scikit-learn", "data science", "predictive analytics", "llm", "fine-tuning",
        "rag", "langchain", "huggingface", "mlflow", "kubeflow", "sagemaker", "vertex ai",
        "databricks", "spark", "ray", "gpu", "cuda", "aws", "gcp", "azure", "kubernetes",
        "docker", "airflow", "ai product", "ml product", "recommendation system",
        "personalization", "ai strategy", "ai transformation", "ai governance",
        "responsible ai", "ai ethics", "ai roadmap", "statistical modeling"
    ]

    non_ai_keywords = [
        "java", "javascript", "react", "angular", "node.js", "python", "sql", "aws", "azure",
        "microservices", "agile", "scrum", "cloud", "api", "database", "ui/ux", "frontend", "backend"
    ]

    for i in range(num_postings):
        is_ai_role = random.random() < 0.7 # 70% chance of being an AI-related role
        title = random.choice(ai_titles) if is_ai_role else random.choice(non_ai_titles)
        company = fake.company()
        location = fake.city()

        description_parts = [fake.paragraph(nb_sentences=5) for _ in range(random.randint(2, 5))]
        description = " ".join(description_parts)

        # Inject AI/non-AI keywords based on role type
        if is_ai_role:
            num_ai_skills = random.randint(3, 10)
            skills_to_add = random.sample(ai_keywords, num_ai_skills)
            description = description + " " + ", ".join(skills_to_add) + ". " + fake.text(max_nb_chars=200)
        else:
            num_non_ai_skills = random.randint(3, 7)
            skills_to_add = random.sample(non_ai_keywords, num_non_ai_skills)
            description = description + " " + ", ".join(skills_to_add) + ". " + fake.text(max_nb_chars=150)

        # Add some randomness to ensure case-insensitivity tests work
        if random.random() < 0.3:
            description = description.replace("python", "Python").replace("tensorflow", "TensorFlow")
            title = title.replace("engineer", "Engineer")

        source = random.choice(["LinkedIn", "Indeed", "Company Careers Page"])
        url = fake.url()
        posted_date = (datetime.now() - timedelta(days=random.randint(0, 90))).strftime('%Y-%m-%d') # Last 90 days

        job_postings.append({
            "title": title,
            "description": description,
            "company": company,
            "location": location,
            "source": source,
            "url": url,
            "posted_date": posted_date
        })

    return job_postings

# Execute function to generate job postings
raw_job_postings = generate_synthetic_job_postings(num_postings=2000) # Generate more than 1500 to allow for deduplication
print(f"Generated {len(raw_job_postings)} raw job postings.")
print("\nSample Job Posting:")
print(raw_job_postings[0])
```

The output above confirms that our simulated data generation is working, providing a base list of job postings for further analysis. This is the raw material that your HR team needs you to process into actionable insights.

---

## 2. Defining AI Talent Taxonomy and Seniority Classification Logic

The HR team needs to categorize job roles efficiently. Your next task is to implement the core logic for identifying AI-related skills and classifying job seniority. This is crucial for understanding the skill landscape and the experience levels being sought in the market. This stage directly uses the predefined AI skill taxonomy and seniority indicators from the HR requirements.

### 2.1 Define AI Skill Taxonomy

The AI skill taxonomy is a comprehensive dictionary mapping categories of AI to specific keywords. This structure helps in systematically identifying diverse AI competencies within job descriptions. You'll define the `SkillCategory` Enum and the `AI_SKILLS` dictionary as specified.

```python
class SkillCategory(str, Enum):
    ML_ENGINEERING = "ml_engineering"
    DATA_SCIENCE = "data_science"
    AI_INFRASTRUCTURE = "ai_infrastructure"
    AI_PRODUCT = "ai_product"
    AI_STRATEGY = "ai_strategy"

AI_SKILLS: Dict[SkillCategory, Set[str]] = {
    SkillCategory.ML_ENGINEERING: {
        "pytorch", "tensorflow", "keras", "mlops", "ml engineering",
        "model deployment", "feature engineering", "model training",
        "deep learning", "neural networks", "transformers", "llm",
        "fine-tuning", "rag", "langchain", "huggingface", "machine learning engineering"
    },
    SkillCategory.DATA_SCIENCE: {
        "machine learning", "data science", "statistical modeling",
        "predictive analytics", "a/b testing", "experimentation",
        "python", "r", "sql", "pandas", "numpy", "scikit-learn",
        "data analytics", "data mining"
    },
    SkillCategory.AI_INFRASTRUCTURE: {
        "mlflow", "kubeflow", "sagemaker", "vertex ai", "databricks",
        "spark", "ray", "dask", "gpu", "cuda", "aws", "gcp", "azure",
        "kubernetes", "docker", "airflow", "kafka", "hadoop"
    },
    SkillCategory.AI_PRODUCT: {
        "ai product", "ml product", "product manager ai",
        "chatbot", "recommendation system", "personalization",
        "computer vision", "nlp", "natural language processing",
        "generative ai", "user experience (ai)", "product strategy (ai)"
    },
    SkillCategory.AI_STRATEGY: {
        "ai strategy", "ai transformation", "ai governance",
        "responsible ai", "ai ethics", "ai roadmap", "ai policy",
        "innovation management (ai)"
    }
}

# Example usage (demonstrating the structure)
print(f"Number of AI skill categories: {len(AI_SKILLS)}")
print(f"Skills in ML Engineering: {list(AI_SKILLS[SkillCategory.ML_ENGINEERING])[:5]}...")
```

The `AI_SKILLS` dictionary, with its structured categories, provides a clear framework for extracting specific AI competencies from job descriptions. This allows for a granular understanding of the demand for different types of AI expertise.

### 2.2 Define Seniority Levels and Classification Logic

To help HR understand the experience level of AI professionals being sought, you need to define job seniority levels and a function to classify them based on job titles. This function will be crucial for segmenting the talent market by experience. Remember to handle case-insensitivity for robustness.

```python
class SeniorityLevel(str, Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    DIRECTOR = "director"
    VP = "vp"
    EXECUTIVE = "executive"

SENIORITY_INDICATORS: Dict[SeniorityLevel, List[str]] = {
    SeniorityLevel.ENTRY: ["junior", "entry", "associate", "intern", "graduate", "new grad"],
    SeniorityLevel.MID: ["mid", "intermediate", "ii", "2", "staff"],
    SeniorityLevel.SENIOR: ["senior", "sr", "iii", "3", "experienced"],
    SeniorityLevel.LEAD: ["lead", "principal", "iv", "4", "group"],
    SeniorityLevel.DIRECTOR: ["director", "head of", "mgr"],
    SeniorityLevel.VP: ["vp", "vice president"],
    SeniorityLevel.EXECUTIVE: ["chief", "cto", "cdo", "cao", "evp", "svp"]
}

def classify_seniority(title: str) -> SeniorityLevel:
    """
    Classifies job title seniority based on predefined indicators.
    Performs case-insensitive matching.

    Args:
        title (str): The job title.

    Returns:
        SeniorityLevel: The classified seniority level, defaults to MID if no match.
    """
    title_lower = title.lower()
    # Iterate in reverse order to prioritize higher seniority levels if keywords overlap
    for level in reversed(list(SeniorityLevel)):
        if any(ind in title_lower for ind in SENIORITY_INDICATORS[level]):
            return level
    return SeniorityLevel.MID # Default to Mid-Level if no specific indicators found

# Example usage
print(f"Classification for 'Junior ML Engineer': {classify_seniority('Junior ML Engineer')}")
print(f"Classification for 'Principal Data Scientist': {classify_seniority('Principal Data Scientist')}")
print(f"Classification for 'Data Scientist II': {classify_seniority('Data Scientist II')}")
print(f"Classification for 'VP of AI Strategy': {classify_seniority('VP of AI Strategy')}")
print(f"Classification for 'AI Engineer': {classify_seniority('AI Engineer')}")
```

The `classify_seniority` function, by providing a standardized seniority level, allows the HR team to analyze demand across different experience brackets. This is vital for understanding talent supply and demand imbalances, and for tailoring recruitment efforts to specific experience profiles.

### 2.3 Implement AI Skill Extraction

Based on the `AI_SKILLS` taxonomy, you need a function to extract all relevant AI skills from a job description. This function will be applied to every job posting to build a comprehensive view of desired skills. Robustness here means ensuring case-insensitive matching.

```python
def extract_ai_skills(text: str) -> Set[str]:
    """
    Extracts AI skills from a job description text, performing case-insensitive matching.

    Args:
        text (str): The full job description text.

    Returns:
        Set[str]: A set of unique AI skills found in the text.
    """
    text_lower = text.lower()
    found_skills = set()
    for category, skills_set in AI_SKILLS.items():
        for skill in skills_set:
            if skill in text_lower:
                found_skills.add(skill)
    return found_skills

# Example usage
sample_description = "We are looking for a deep learning engineer with experience in PyTorch, Transformers, and MLOps. Strong Python skills and familiar with AWS infrastructure."
extracted = extract_ai_skills(sample_description)
print(f"\nSkills extracted from sample description: {extracted}")

sample_description_case_issue = "Candidate must have Pytorch experience and be proficient in TENSORFLOW."
extracted_case_issue = extract_ai_skills(sample_description_case_issue)
print(f"Skills extracted with case-insensitivity: {extracted_case_issue}")
```

The `extract_ai_skills` function is a cornerstone of this tool. By accurately identifying AI-related skills, HR can determine which specific competencies are most in demand, guiding training programs and recruitment focus. The case-insensitive matching prevents missing valuable data due to minor text variations.

---

## 3. Calculating the AI Relevance Score

To quantify how "AI-focused" each job role is, the HR team requires an **AI Relevance Score**. This score will help prioritize jobs, filter for highly specialized AI roles, and understand the depth of AI integration in various positions. Your task is to implement this scoring mechanism using the specified formula.

### 3.1 Implement AI Relevance Scoring Function

The AI relevance score combines the number of extracted AI skills with the explicit mention of AI-specific keywords in the job title. This provides a balanced view, considering both the depth of skill requirements and the role's explicit AI focus. The formula is designed to give more weight to the presence of skills while boosting for explicit AI terms in the title.

The formula for the AI Relevance Score (ranging from 0 to 1) is:
$$ \text{min}(\frac{\text{len(skills)}}{5}, 1.0) * 0.6 + (0.4 \text{ if AI keywords present in title else } 0.0) $$

Where:
*   $ \text{len(skills)} $ is the number of unique AI skills extracted from the job description.
*   The division by $5$ normalizes the skill count to a maximum of $1.0$ if 5 or more unique skills are found.
*   The $0.6$ and $0.4$ weights represent the contribution of skills and title keywords, respectively.
*   AI keywords in title are `["ai", "ml", "machine learning", "data scientist", "mlops"]`.

```python
def calculate_ai_relevance_score(skills: Set[str], title: str) -> float:
    """
    Calculates a 0-1 AI relevance score for a job posting.

    The score combines the number of extracted AI skills from the description
    and the presence of AI-specific keywords in the job title.

    Args:
        skills (Set[str]): A set of extracted AI skills.
        title (str): The job title.

    Returns:
        float: The calculated AI relevance score (0.0 to 1.0).
    """
    # Base score from the number of extracted skills (max 1.0 if 5+ skills)
    base_score = min(len(skills) / 5, 1.0) * 0.6

    # Boost score if AI-specific keywords are present in the title
    title_lower = title.lower()
    title_keywords = ["ai", "ml", "machine learning", "data scientist", "mlops", "artificial intelligence"]
    title_boost = 0.4 if any(kw in title_lower for kw in title_keywords) else 0.0

    # Combine scores, ensuring it doesn't exceed 1.0
    return min(base_score + title_boost, 1.0)

# Example usage
example_title_1 = "Senior Machine Learning Engineer"
example_skills_1 = {"pytorch", "mlops", "deep learning", "transformers", "aws"} # 5 skills
score_1 = calculate_ai_relevance_score(example_skills_1, example_title_1)
print(f"Title: '{example_title_1}', Skills: {example_skills_1}")
print(f"AI Relevance Score: {score_1:.2f} (Base: {min(len(example_skills_1)/5, 1.0)*0.6:.2f}, Title Boost: {0.4 if any(kw in example_title_1.lower() for kw in ['ai', 'ml', 'machine learning', 'data scientist', 'mlops', 'artificial intelligence']) else 0.0:.1f})\n")

example_title_2 = "Software Engineer"
example_skills_2 = {"python", "cloud"} # Not AI skills as per taxonomy
score_2 = calculate_ai_relevance_score(example_skills_2, example_title_2)
print(f"Title: '{example_title_2}', Skills: {example_skills_2}")
print(f"AI Relevance Score: {score_2:.2f} (Base: {min(len(example_skills_2)/5, 1.0)*0.6:.2f}, Title Boost: {0.4 if any(kw in example_title_2.lower() for kw in ['ai', 'ml', 'machine learning', 'data scientist', 'mlops', 'artificial intelligence']) else 0.0:.1f})\n")

example_title_3 = "Data Scientist"
example_skills_3 = {"scikit-learn", "statistical modeling", "python"} # 3 AI skills
score_3 = calculate_ai_relevance_score(example_skills_3, example_title_3)
print(f"Title: '{example_title_3}', Skills: {example_skills_3}")
print(f"AI Relevance Score: {score_3:.2f} (Base: {min(len(example_skills_3)/5, 1.0)*0.6:.2f}, Title Boost: {0.4 if any(kw in example_title_3.lower() for kw in ['ai', 'ml', 'machine learning', 'data scientist', 'mlops', 'artificial intelligence']) else 0.0:.1f})\n")
```

The AI Relevance Score function provides a quantifiable metric that HR can use to quickly identify and prioritize highly AI-centric roles. This helps them filter noise and focus recruitment efforts on positions that truly align with the company's AI strategy. A score closer to 1.0 indicates a deeply integrated AI role, while a lower score might suggest a more generalist role with some AI exposure.

---

## 4. Processing and Deduplicating Job Postings

Now that all the individual logic components are defined, it's time to apply them to the entire dataset of raw job postings. A critical step in real-world data processing is **deduplication** to ensure data quality and prevent overcounting. You will process each job, extract features, and then clean the dataset.

### 4.1 Process Raw Job Postings

This step involves iterating through all generated job postings and applying the `extract_ai_skills`, `classify_seniority`, and `calculate_ai_relevance_score` functions. The results will be stored in a structured format suitable for further analysis.

```python
def process_job_postings(raw_jobs: List[Dict]) -> List[Dict]:
    """
    Processes raw job postings to extract skills, classify seniority,
    and calculate AI relevance scores.

    Args:
        raw_jobs (List[Dict]): A list of raw job posting dictionaries.

    Returns:
        List[Dict]: A list of processed job posting dictionaries with added features.
    """
    processed_jobs = []
    for job in raw_jobs:
        title = job['title']
        description = job['description']

        # Extract AI skills
        extracted_skills = extract_ai_skills(description)

        # Classify seniority
        seniority_level = classify_seniority(title)

        # Calculate AI relevance score
        ai_relevance_score = calculate_ai_relevance_score(extracted_skills, title)

        processed_job = job.copy()
        processed_job['extracted_skills'] = list(extracted_skills) # Convert set to list for easier storage
        processed_job['seniority_level'] = seniority_level.value
        processed_job['ai_relevance_score'] = ai_relevance_score
        processed_jobs.append(processed_job)
    return processed_jobs

# Execute processing
processed_job_postings = process_job_postings(raw_job_postings)
print(f"Processed {len(processed_job_postings)} job postings.")
print("\nSample Processed Job Posting:")
print(processed_job_postings[0])
```

Processing the raw job postings transforms unstructured text into structured features. This step is essential for converting raw data into a format that the HR team can analyze, providing a rich set of attributes for each job posting.

### 4.2 Deduplicate Processed Job Postings

Job postings from multiple sources can often be duplicates. To ensure accurate market analysis, it's crucial to identify and remove these redundant entries. The deduplication strategy involves creating a unique hash for each job based on a combination of its title, company, and location. This allows for robust identification of identical jobs.

```python
def deduplicate_job_postings(jobs: List[Dict]) -> List[Dict]:
    """
    Deduplicates a list of job postings based on a hash of title, company, and location.

    Args:
        jobs (List[Dict]): A list of job posting dictionaries.

    Returns:
        List[Dict]: A list of unique job posting dictionaries.
    """
    unique_jobs = []
    seen_hashes = set()

    for job in jobs:
        # Create a unique key for deduplication
        # Normalizing title, company, and location to lower case and removing extra spaces
        dedup_key = f"{job['title'].lower().strip()}-{job['company'].lower().strip()}-{job['location'].lower().strip()}"
        job_hash = hashlib.md5(dedup_key.encode('utf-8')).hexdigest()

        if job_hash not in seen_hashes:
            unique_jobs.append(job)
            seen_hashes.add(job_hash)
    return unique_jobs

# Execute deduplication
deduplicated_job_postings = deduplicate_job_postings(processed_job_postings)
print(f"Initial processed postings: {len(processed_job_postings)}")
print(f"Deduplicated postings: {len(deduplicated_job_postings)}")
print(f"Removed {len(processed_job_postings) - len(deduplicated_job_postings)} duplicates.")

# Convert to Pandas DataFrame for easier manipulation
df_jobs = pd.DataFrame(deduplicated_job_postings)
df_jobs['posted_date'] = pd.to_datetime(df_jobs['posted_date'])
print("\nFirst 5 rows of the final DataFrame:")
print(df_jobs.head())
```

Deduplication is a vital data quality step. By eliminating duplicate entries, you ensure that the subsequent analysis accurately reflects the actual market demand, rather than being skewed by multiple listings of the same job. This gives HR a more reliable view of the talent landscape. The resulting `df_jobs` DataFrame is now clean and ready for in-depth analysis.

---

## 5. Temporal Aggregation for Trend Analysis

The HR and strategy teams need to track how demand for AI talent evolves over time. Your next task is to aggregate the processed job data on a weekly basis. This will enable time-series analysis of job volumes, popular skills, and demand distribution across seniority levels, providing crucial insights into market dynamics.

### 5.1 Aggregate Job Postings by Week

To understand trends, we need to group job postings by the week they were posted. This involves calculating the start of the week for each `posted_date` and then counting the number of jobs for each week.

```python
def aggregate_weekly_job_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the number of job postings on a weekly basis.

    Args:
        df (pd.DataFrame): DataFrame containing job postings with a 'posted_date' column.

    Returns:
        pd.DataFrame: DataFrame with 'week_start' and 'job_count' columns.
    """
    df_copy = df.copy()
    df_copy['week_start'] = df_copy['posted_date'].dt.to_period('W').dt.start_time
    weekly_job_volume = df_copy.groupby('week_start').size().reset_index(name='job_count')
    return weekly_job_volume.sort_values('week_start')

# Execute weekly job volume aggregation
weekly_job_volume_df = aggregate_weekly_job_volume(df_jobs)
print("Weekly Job Volume (first 5 weeks):")
print(weekly_job_volume_df.head())
```

Aggregating job volume by week provides HR with a clear trend line, showing whether demand for AI talent is increasing or decreasing. This information is invaluable for capacity planning in recruitment and for understanding the overall health of the AI talent market.

### 5.2 Identify Top Weekly Skills and Seniority Distribution

Beyond just job volume, HR needs to know which skills are most in demand and how seniority distributions are shifting. You will aggregate the most frequently mentioned skills and the distribution of seniority levels on a weekly basis.

```python
def aggregate_weekly_skills_and_seniority(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates top skills and seniority level distribution on a weekly basis.

    Args:
        df (pd.DataFrame): DataFrame with 'posted_date', 'extracted_skills', and 'seniority_level' columns.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame with weekly top skills ('week_start', 'skill', 'count').
            - DataFrame with weekly seniority distribution ('week_start', 'seniority_level', 'count').
    """
    df_copy = df.copy()
    df_copy['week_start'] = df_copy['posted_date'].dt.to_period('W').dt.start_time

    # Aggregate skills
    weekly_skills_data = []
    for week_start, group in df_copy.groupby('week_start'):
        all_skills_this_week = [skill for sublist in group['extracted_skills'] for skill in sublist]
        skill_counts = Counter(all_skills_this_week)
        for skill, count in skill_counts.most_common(10): # Top 10 skills per week
            weekly_skills_data.append({'week_start': week_start, 'skill': skill, 'count': count})
    weekly_top_skills_df = pd.DataFrame(weekly_skills_data)

    # Aggregate seniority
    weekly_seniority_df = df_copy.groupby(['week_start', 'seniority_level']).size().reset_index(name='count')
    return weekly_top_skills_df.sort_values(['week_start', 'count'], ascending=[True, False]), weekly_seniority_df.sort_values('week_start')

# Execute weekly skills and seniority aggregation
weekly_top_skills_df, weekly_seniority_df = aggregate_weekly_skills_and_seniority(df_jobs)

print("\nWeekly Top Skills (first 5 entries):")
print(weekly_top_skills_df.head())
print("\nWeekly Seniority Distribution (first 5 entries):")
print(weekly_seniority_df.head())
```

These aggregations provide HR with granular insights. Tracking weekly top skills helps identify emerging technologies or shifting demand for specific competencies. Analyzing seniority distribution over time reveals if the market is favoring junior, mid-level, or senior talent, allowing HR to adjust their recruitment campaigns and compensation strategies accordingly.

---

## 6. Visualizing AI Talent Market Insights

The final step in your workflow is to present these complex insights in a clear, actionable format for the HR and strategy teams. Visualizations make it easy to grasp trends and make data-driven decisions. You will generate charts for weekly job posting trends, top 10 overall requested skills, and the distribution of demand across seniority levels.

### 6.1 Visualize Weekly Trends in AI Job Postings

This visualization will show the overall volume of AI job postings over time, helping HR understand the general trajectory of the AI talent market.

```python
def plot_weekly_job_volume(df_weekly_volume: pd.DataFrame):
    """
    Generates a line plot showing weekly trends in AI job posting volume.

    Args:
        df_weekly_volume (pd.DataFrame): DataFrame with 'week_start' and 'job_count'.
    """
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_weekly_volume, x='week_start', y='job_count', marker='o')
    plt.title('Weekly Trend in AI Job Postings Volume', fontsize=16)
    plt.xlabel('Week Start Date', fontsize=12)
    plt.ylabel('Number of Job Postings', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Execute visualization
plot_weekly_job_volume(weekly_job_volume_df)
```

This chart provides a quick and clear overview of market activity. A rising trend indicates a hot market, requiring faster recruitment, while a flat or declining trend might suggest opportunities for more strategic sourcing or a need to re-evaluate internal skill development.

### 6.2 Visualize Top 10 Most Requested AI Skills

Understanding which specific AI skills are most sought after is critical for talent development and acquisition strategies. This bar chart will highlight the overall top 10 skills across the entire collected period.

```python
def plot_top_skills(df_jobs: pd.DataFrame):
    """
    Generates a bar chart showing the top 10 most requested AI skills.

    Args:
        df_jobs (pd.DataFrame): DataFrame containing job postings with 'extracted_skills'.
    """
    all_skills = [skill for sublist in df_jobs['extracted_skills'] for skill in sublist]
    skill_counts = Counter(all_skills)
    top_10_skills = pd.DataFrame(skill_counts.most_common(10), columns=['skill', 'count'])

    plt.figure(figsize=(14, 7))
    sns.barplot(x='count', y='skill', data=top_10_skills, palette='viridis')
    plt.title('Top 10 Most Requested AI Skills', fontsize=16)
    plt.xlabel('Frequency (Number of Job Postings Mentioning Skill)', fontsize=12)
    plt.ylabel('AI Skill', fontsize=12)
    plt.tight_layout()
    plt.show()

# Execute visualization
plot_top_skills(df_jobs)
```

This visualization directly informs HR about which skills are currently commanding the most market attention. This intelligence can guide decisions on internal training programs, university partnerships, and targeted marketing for job advertisements.

### 6.3 Visualize AI Job Demand Distribution Across Seniority Levels

Analyzing the distribution of demand across different seniority levels reveals where the most significant talent gaps or competitive pressures might exist.

```python
def plot_seniority_distribution(df_jobs: pd.DataFrame):
    """
    Generates a bar chart showing the distribution of AI job demand across seniority levels.

    Args:
        df_jobs (pd.DataFrame): DataFrame containing job postings with 'seniority_level'.
    """
    seniority_order = [level.value for level in SeniorityLevel]
    seniority_counts = df_jobs['seniority_level'].value_counts().reindex(seniority_order, fill_value=0)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=seniority_counts.index, y=seniority_counts.values, palette='coolwarm')
    plt.title('Distribution of AI Job Demand by Seniority Level', fontsize=16)
    plt.xlabel('Seniority Level', fontsize=12)
    plt.ylabel('Number of Job Postings', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Execute visualization
plot_seniority_distribution(df_jobs)
```

This chart helps HR understand whether the market is saturated with junior roles, desperately seeking senior leaders, or balanced across experience levels. This insight is critical for tailoring recruitment campaigns and understanding the competitive landscape for specific talent segments. For example, a high demand for "Lead" or "Director" level positions might signal an executive talent gap.

---

## Conclusion

Congratulations! As a Software Developer, you have successfully built the foundational components of an internal AI Talent Demand Tracker. You've navigated through simulating data acquisition, implemented sophisticated text processing for skill extraction and seniority classification, developed a quantitative AI relevance score, performed crucial data cleaning through deduplication, and aggregated insights for temporal analysis. Finally, you've transformed complex data into actionable visualizations that directly address the strategic needs of your HR and strategy teams.

This tool provides your organization with a dynamic way to understand the evolving demand for AI talent, enabling proactive adaptation of hiring strategies and a focused approach to identifying emerging skill trends. Your work directly contributes to the company's ability to stay agile and competitive in the rapidly changing landscape of AI talent.
