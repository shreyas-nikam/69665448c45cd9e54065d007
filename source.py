# --- UPDATED imports (remove playwright/httpx/bs4/asyncio/time related scraping deps) ---
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

# NEW: python-jobspy
# pip install -U python-jobspy :contentReference[oaicite:0]{index=0}
from jobspy import scrape_jobs

# Configure plot styles for better readability
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

# ============================================================================
# Web Scraping Functions (REPLACED with python-jobspy)
# ============================================================================

# JobSpy expects site_name values like: linkedin, indeed, glassdoor, google, zip_recruiter :contentReference[oaicite:1]{index=1}
_SOURCE_TO_JOBSPY_SITE = {
    "linkedin": "linkedin",
    "indeed": "indeed",
    "glassdoor": "glassdoor",
    "google": "google",
    "ziprecruiter": "zip_recruiter",
    "zip_recruiter": "zip_recruiter",
    "bayt": "bayt",
    "naukri": "naukri",
    "bdjobs": "bdjobs",
}


def _safe_str(x) -> str:
    return "" if x is None else str(x)


def _normalize_posted_date(val) -> str:
    """
    JobSpy commonly returns a `date_posted` column (string or datetime-like). :contentReference[oaicite:2]{index=2}
    Normalize to 'YYYY-MM-DD'. If missing/unparseable, default to today.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return datetime.now().strftime("%Y-%m-%d")
    try:
        dt = pd.to_datetime(val, errors="coerce")
        if pd.isna(dt):
            return datetime.now().strftime("%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def _jobspy_df_to_jobs(df: pd.DataFrame) -> List[Dict]:
    """
    Converts JobSpy dataframe to the schema used throughout this file:
      title, company, location, description, source, url, posted_date

    JobSpy output is commonly lowercase (e.g., site/title/company/location/job_url/description/date_posted). :contentReference[oaicite:3]{index=3}
    Some examples/docs show uppercase columns (SITE/TITLE/COMPANY/CITY/STATE/JOB_URL/DESCRIPTION). :contentReference[oaicite:4]{index=4}
    This adapter supports both.
    """
    if df is None or df.empty:
        return []

    # Build a case-insensitive column map
    cols = {c.lower(): c for c in df.columns}

    def col(*names: str):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_site = col("site", "SITE")
    c_title = col("title", "TITLE")
    c_company = col("company", "COMPANY")
    c_location = col("location", "LOCATION")
    c_city = col("city", "CITY")
    c_state = col("state", "STATE")
    c_url = col("job_url", "JOB_URL", "job_url_direct", "JOB_URL_DIRECT")
    c_desc = col("description", "DESCRIPTION")
    c_date = col("date_posted", "DATE_POSTED")

    jobs: List[Dict] = []
    for _, row in df.iterrows():
        # Location: prefer single `location`; else "City, State"
        if c_location:
            location = _safe_str(row.get(c_location)).strip()
        else:
            city = _safe_str(row.get(c_city)).strip() if c_city else ""
            state = _safe_str(row.get(c_state)).strip() if c_state else ""
            location = ", ".join(
                [p for p in [city, state] if p]) or "Not specified"

        title = _safe_str(row.get(c_title)).strip() if c_title else ""
        company = _safe_str(row.get(c_company)).strip(
        ) if c_company else "Unknown Company"
        description = _safe_str(row.get(c_desc)).strip() if c_desc else ""
        if not description:
            description = f"Job posting for {title} at {company}".strip()

        jobs.append(
            {
                "title": title or "Unknown Title",
                "company": company,
                "location": location,
                "description": description,
                "source": _safe_str(row.get(c_site)).strip().title() if c_site else "JobSpy",
                "url": _safe_str(row.get(c_url)).strip() if c_url else "",
                "posted_date": _normalize_posted_date(row.get(c_date) if c_date else None),
            }
        )

    return jobs


def scrape_jobs_from_multiple_sources(
    search_query: str,
    sources: List[str] = ["linkedin", "indeed", "glassdoor"],
    max_results_per_source: int = 50,
    location: str = "United States",
    # ~last 30 days; JobSpy supports hours_old in scrape_jobs :contentReference[oaicite:5]{index=5}
    hours_old: int = 24 * 30,
    country_indeed: str = "USA",
    linkedin_fetch_description: bool = True,
) -> List[Dict]:
    """
    Scrapes job postings using python-jobspy and returns the same List[Dict] schema
    used by the rest of this file.

    Notes:
    - JobSpy aggregates multiple sites into one dataframe. :contentReference[oaicite:6]{index=6}
    - Output includes common fields like title/company/location/site/description/date_posted/job_url. :contentReference[oaicite:7]{index=7}
    """
    # Map requested sources to jobspy site_name values
    site_name: List[str] = []
    for s in (sources or []):
        key = s.strip().lower()
        if key in _SOURCE_TO_JOBSPY_SITE:
            site_name.append(_SOURCE_TO_JOBSPY_SITE[key])

    # Default to the same trio if list becomes empty
    if not site_name:
        site_name = ["linkedin", "indeed", "glassdoor"]

    # JobSpy returns a DataFrame
    # results_wanted is total results across sites (not per-site), so request len(site_name)*max_results_per_source :contentReference[oaicite:8]{index=8}
    df = scrape_jobs(
        site_name=site_name,
        search_term=search_query,
        location=location,
        results_wanted=max_results_per_source * len(site_name),
        hours_old=hours_old,
        country_indeed=country_indeed,
        linkedin_fetch_description=linkedin_fetch_description,
    )

    return _jobspy_df_to_jobs(df)

# ============================================================================
# Original Synthetic Data Generation Function (kept as fallback)
# ============================================================================


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
        "python",
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
        "java", "javascript", "react", "angular", "node.js", "sql", "aws", "azure",
        "microservices", "agile", "scrum", "cloud", "api", "database", "ui/ux", "frontend", "backend"
    ]

    for i in range(num_postings):
        is_ai_role = random.random() < 0.7  # 70% chance of being an AI-related role
        title = random.choice(
            ai_titles) if is_ai_role else random.choice(non_ai_titles)
        company = fake.company()
        location = fake.city()

        description_parts = [fake.paragraph(
            nb_sentences=5) for _ in range(random.randint(2, 5))]
        description = " ".join(description_parts)

        # Inject AI/non-AI keywords based on role type
        if is_ai_role:
            num_ai_skills = random.randint(3, 10)
            skills_to_add = random.sample(ai_keywords, num_ai_skills)
            description = description + " " + \
                ", ".join(skills_to_add) + ". " + fake.text(max_nb_chars=200)
        else:
            num_non_ai_skills = random.randint(3, 7)
            skills_to_add = random.sample(non_ai_keywords, num_non_ai_skills)
            description = description + " " + \
                ", ".join(skills_to_add) + ". " + fake.text(max_nb_chars=150)

        # Add some randomness to ensure case-insensitivity tests work
        if random.random() < 0.3:
            description = description.replace(
                "python", "Python").replace("tensorflow", "TensorFlow")
            title = title.replace("engineer", "Engineer")

        source = random.choice(["LinkedIn", "Indeed", "Company Careers Page"])
        url = fake.url()
        posted_date = (datetime.now() - timedelta(days=random.randint(0, 90))
                       ).strftime('%Y-%m-%d')  # Last 90 days

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
# Generate more than 1500 to allow for deduplication
# raw_job_postings = generate_synthetic_job_postings(num_postings=2000)
# print(f"Generated {len(raw_job_postings)} raw job postings.")
# print("\nSample Job Posting:")
# print(raw_job_postings[0])


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
        "python", "sql", "pandas", "numpy", "scikit-learn",
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
# print(f"Number of AI skill categories: {len(AI_SKILLS)}")
# print(
#     f"Skills in ML Engineering: {list(AI_SKILLS[SkillCategory.ML_ENGINEERING])[:5]}...")


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
    return SeniorityLevel.MID  # Default to Mid-Level if no specific indicators found


# Example usage
# print(
#     f"Classification for 'Junior ML Engineer': {classify_seniority('Junior ML Engineer')}")
# print(
#     f"Classification for 'Principal Data Scientist': {classify_seniority('Principal Data Scientist')}")
# print(
#     f"Classification for 'Data Scientist II': {classify_seniority('Data Scientist II')}")
# print(
#     f"Classification for 'VP of AI Strategy': {classify_seniority('VP of AI Strategy')}")
# print(f"Classification for 'AI Engineer': {classify_seniority('AI Engineer')}")


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
# sample_description = "We are looking for a deep learning engineer with experience in PyTorch, Transformers, and MLOps. Strong Python skills and familiar with AWS infrastructure."
# extracted = extract_ai_skills(sample_description)
# print(f"\nSkills extracted from sample description: {extracted}")

# sample_description_case_issue = "Candidate must have Pytorch experience and be proficient in TENSORFLOW."
# extracted_case_issue = extract_ai_skills(sample_description_case_issue)
# print(f"Skills extracted with case-insensitivity: {extracted_case_issue}")


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
    title_keywords = ["ai", "ml", "machine learning",
                      "data scientist", "mlops", "artificial intelligence"]
    title_boost = 0.4 if any(
        kw in title_lower for kw in title_keywords) else 0.0

    # Combine scores, ensuring it doesn't exceed 1.0
    return min(base_score + title_boost, 1.0)


# Example usage
# example_title_1 = "Senior Machine Learning Engineer"
# example_skills_1 = {"pytorch", "mlops",
#                     "deep learning", "transformers", "aws"}  # 5 skills
# score_1 = calculate_ai_relevance_score(example_skills_1, example_title_1)
# print(f"Title: '{example_title_1}', Skills: {example_skills_1}")
# print(
#     f"AI Relevance Score: {score_1:.2f} (Base: {min(len(example_skills_1)/5, 1.0)*0.6:.2f}, Title Boost: {0.4 if any(kw in example_title_1.lower() for kw in ['ai', 'ml', 'machine learning', 'data scientist', 'mlops', 'artificial intelligence']) else 0.0:.1f})\n")

# example_title_2 = "Software Engineer"
# example_skills_2 = {"python", "cloud"}  # Not AI skills as per taxonomy
# score_2 = calculate_ai_relevance_score(example_skills_2, example_title_2)
# print(f"Title: '{example_title_2}', Skills: {example_skills_2}")
# print(
#     f"AI Relevance Score: {score_2:.2f} (Base: {min(len(example_skills_2)/5, 1.0)*0.6:.2f}, Title Boost: {0.4 if any(kw in example_title_2.lower() for kw in ['ai', 'ml', 'machine learning', 'data scientist', 'mlops', 'artificial intelligence']) else 0.0:.1f})\n")

# example_title_3 = "Data Scientist"
# example_skills_3 = {"scikit-learn",
#                     "statistical modeling", "python"}  # 3 AI skills
# score_3 = calculate_ai_relevance_score(example_skills_3, example_title_3)
# print(f"Title: '{example_title_3}', Skills: {example_skills_3}")
# print(
#     f"AI Relevance Score: {score_3:.2f} (Base: {min(len(example_skills_3)/5, 1.0)*0.6:.2f}, Title Boost: {0.4 if any(kw in example_title_3.lower() for kw in ['ai', 'ml', 'machine learning', 'data scientist', 'mlops', 'artificial intelligence']) else 0.0:.1f})\n")


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
        ai_relevance_score = calculate_ai_relevance_score(
            extracted_skills, title)

        processed_job = job.copy()
        # Convert set to list for easier storage
        processed_job['extracted_skills'] = list(extracted_skills)
        processed_job['seniority_level'] = seniority_level.value
        processed_job['ai_relevance_score'] = ai_relevance_score
        processed_jobs.append(processed_job)
    return processed_jobs


# Execute processing
# processed_job_postings = process_job_postings(raw_job_postings)
# print(f"Processed {len(processed_job_postings)} job postings.")
# print("\nSample Processed Job Posting:")
# print(processed_job_postings[0])


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
# deduplicated_job_postings = deduplicate_job_postings(processed_job_postings)
# print(f"Initial processed postings: {len(processed_job_postings)}")
# print(f"Deduplicated postings: {len(deduplicated_job_postings)}")
# print(
#     f"Removed {len(processed_job_postings) - len(deduplicated_job_postings)} duplicates.")

# # Convert to Pandas DataFrame for easier manipulation
# df_jobs = pd.DataFrame(deduplicated_job_postings)
# df_jobs['posted_date'] = pd.to_datetime(df_jobs['posted_date'])
# print("\nFirst 5 rows of the final DataFrame:")
# print(df_jobs.head())


def aggregate_weekly_job_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the number of job postings on a weekly basis.

    Args:
        df (pd.DataFrame): DataFrame containing job postings with a 'posted_date' column.

    Returns:
        pd.DataFrame: DataFrame with 'week_start' and 'job_count' columns.
    """
    df_copy = df.copy()
    df_copy['week_start'] = df_copy['posted_date'].dt.to_period(
        'W').dt.start_time
    weekly_job_volume = df_copy.groupby(
        'week_start').size().reset_index(name='job_count')
    return weekly_job_volume.sort_values('week_start')


# Execute weekly job volume aggregation
# weekly_job_volume_df = aggregate_weekly_job_volume(df_jobs)
# print("Weekly Job Volume (first 5 weeks):")
# print(weekly_job_volume_df.head())


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
    df_copy['week_start'] = df_copy['posted_date'].dt.to_period(
        'W').dt.start_time

    # Aggregate skills
    weekly_skills_data = []
    for week_start, group in df_copy.groupby('week_start'):
        all_skills_this_week = [
            skill for sublist in group['extracted_skills'] for skill in sublist]
        skill_counts = Counter(all_skills_this_week)
        # Top 10 skills per week
        for skill, count in skill_counts.most_common(10):
            weekly_skills_data.append(
                {'week_start': week_start, 'skill': skill, 'count': count})

    # Create DataFrame with proper columns even if empty
    if weekly_skills_data:
        weekly_top_skills_df = pd.DataFrame(weekly_skills_data)
        weekly_top_skills_df = weekly_top_skills_df.sort_values(
            ['week_start', 'count'], ascending=[True, False])
    else:
        weekly_top_skills_df = pd.DataFrame(
            columns=['week_start', 'skill', 'count'])

    # Aggregate seniority
    weekly_seniority_df = df_copy.groupby(
        ['week_start', 'seniority_level']).size().reset_index(name='count')
    weekly_seniority_df = weekly_seniority_df.sort_values('week_start')

    return weekly_top_skills_df, weekly_seniority_df


# Execute weekly skills and seniority aggregation
# weekly_top_skills_df, weekly_seniority_df = aggregate_weekly_skills_and_seniority(
#     df_jobs)

# print("\nWeekly Top Skills (first 5 entries):")
# print(weekly_top_skills_df.head())
# print("\nWeekly Seniority Distribution (first 5 entries):")
# print(weekly_seniority_df.head())


def plot_weekly_job_volume(df_weekly_volume: pd.DataFrame):
    """
    Generates a line plot showing weekly trends in AI job posting volume.

    Args:
        df_weekly_volume (pd.DataFrame): DataFrame with 'week_start' and 'job_count'.
    """
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_weekly_volume, x='week_start',
                 y='job_count', marker='o')
    plt.title('Weekly Trend in AI Job Postings Volume', fontsize=16)
    plt.xlabel('Week Start Date', fontsize=12)
    plt.ylabel('Number of Job Postings', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Execute visualization
# plot_weekly_job_volume(weekly_job_volume_df)


def plot_top_skills(df_jobs: pd.DataFrame):
    """
    Generates a bar chart showing the top 10 most requested AI skills.

    Args:
        df_jobs (pd.DataFrame): DataFrame containing job postings with 'extracted_skills'.
    """
    all_skills = [skill for sublist in df_jobs['extracted_skills']
                  for skill in sublist]
    skill_counts = Counter(all_skills)
    top_10_skills = pd.DataFrame(
        skill_counts.most_common(10), columns=['skill', 'count'])

    plt.figure(figsize=(14, 7))
    sns.barplot(x='count', y='skill', data=top_10_skills, palette='viridis')
    plt.title('Top 10 Most Requested AI Skills', fontsize=16)
    plt.xlabel('Frequency (Number of Job Postings Mentioning Skill)', fontsize=12)
    plt.ylabel('AI Skill', fontsize=12)
    plt.tight_layout()
    plt.show()


# Execute visualization
# plot_top_skills(df_jobs)


def plot_seniority_distribution(df_jobs: pd.DataFrame):
    """
    Generates a bar chart showing the distribution of AI job demand across seniority levels.

    Args:
        df_jobs (pd.DataFrame): DataFrame containing job postings with 'seniority_level'.
    """
    seniority_order = [level.value for level in SeniorityLevel]
    seniority_counts = df_jobs['seniority_level'].value_counts().reindex(
        seniority_order, fill_value=0)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=seniority_counts.index,
                y=seniority_counts.values, palette='coolwarm')
    plt.title('Distribution of AI Job Demand by Seniority Level', fontsize=16)
    plt.xlabel('Seniority Level', fontsize=12)
    plt.ylabel('Number of Job Postings', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# Execute visualization
# plot_seniority_distribution(df_jobs)
