id: 69665448c45cd9e54065d007_documentation
summary: Job Signals & Talent Data Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Building an AI Talent Demand Tracker

## 1. Introduction and Data Generation
Duration: 10:00

Welcome to your next lab for building the "PE-Org AIR System"! Your mission is to empower the HR and strategy teams with crucial insights into the evolving demand for AI talent in the broader market. The goal is to build an internal tool that provides actionable intelligence, enabling proactive adaptation of hiring strategies, focus on in-demand skills, and a clear understanding of the competitive landscape for AI professionals. This tool is vital for the company to stay agile and competitive in a fast-changing talent market.

Today, you'll be developing the core components of this system. This involves simulating the data collection, implementing sophisticated logic for extracting AI-related skills and classifying job seniority, calculating a critical AI relevance score, and finally, aggregating this data for time-series analysis and visualization. Every step you take is designed to directly address a business need, transforming raw data into strategic insights for your organization.

### Learning Objectives

By the end of this codelab, you will have implemented a system that:
*   Collects (simulated) job posting data, handling real-world issues like deduplication.
*   Applies a comprehensive AI skill taxonomy to extract relevant skills from job descriptions.
*   Classifies job seniority levels based on job titles.
*   Calculates an AI relevance score for each job posting using a predefined formula.
*   Aggregates data weekly to enable time-series analysis of AI talent market trends.
*   Visualizes key insights, such as weekly job volume, top skills, and seniority distribution.

### 1.1 Install Required Libraries

As a Software Developer, the first step is always to ensure your environment is ready and that you have data to work with. Since live scraping can be complex and rate-limited within a single notebook, we'll simulate the data acquisition phase by generating a diverse set of synthetic job postings that mirror real-world data. This allows us to focus on the processing and analysis logic, which is your primary task.

```console
!pip install pandas faker matplotlib seaborn numpy
```

### 1.2 Import Dependencies

Next, import all the Python libraries and modules that will be used throughout the notebook. This makes sure all necessary functionalities are available from the start.

<aside class="positive">
<b>Note:</b> The `source` module is assumed to contain helper functions like `generate_synthetic_job_postings`, `process_job_postings`, `deduplicate_job_postings`, aggregation functions, plotting functions, and the definitions for `AI_SKILLS`, `SENIORITY_INDICATORS`, `SkillCategory`, and `SeniorityLevel` as described in later sections.
</aside>

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
from io import BytesIO # Needed if saving plots to buffer, but for codelab, we just describe them.
from source import * # Assuming helper functions and definitions are in source.py

# Configure plot styles for better readability
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')
```

### 1.3 Simulate Job Posting Data Collection

The HR team needs a robust dataset to analyze. Your task is to generate at least 1,500 unique job postings, each with a title, a full description, the source, a URL, and a realistic `posted_date`. This synthetic dataset will serve as the foundation for all subsequent analysis, simulating the output of a job scraper.

Executing the data generation function (e.g., `generate_synthetic_job_postings` from `source.py`) will create a list of dictionaries, each representing a raw job posting.

```python
raw_job_postings = generate_synthetic_job_postings(num_postings=2000)
print(f"Generated {len(raw_job_postings)} raw job postings.")
print("\nSample Raw Job Posting:")
import json
print(json.dumps(raw_job_postings[0], indent=2))
```

A sample of the generated raw job posting data would look like this:

```json
{
  "title": "Lead ML Engineer",
  "description": "Seeking an experienced Lead ML Engineer to design, develop, and deploy machine learning models. Must have expertise in Python, TensorFlow, Kubernetes, and MLOps practices. Lead a team of data scientists and engineers.",
  "source": "LinkedIn",
  "url": "https://www.linkedin.com/jobs/123456789",
  "posted_date": "2023-01-15"
}
```

The output above confirms that our simulated data generation is working, providing a base list of job postings for further analysis. This is the raw material that your HR team needs you to process into actionable insights.

## 2. Defining AI Talent Taxonomy and Seniority Classification Logic
Duration: 15:00

The HR team needs to categorize job roles efficiently. Your next task is to implement the core logic for identifying AI-related skills and classifying job seniority. This is crucial for understanding the skill landscape and the experience levels being sought in the market. This stage directly uses the predefined AI skill taxonomy and seniority indicators from the HR requirements.

### 2.1 Define AI Skill Taxonomy

The AI skill taxonomy is a comprehensive dictionary mapping categories of AI to specific keywords. This structure helps in systematically identifying diverse AI competencies within job descriptions. You'll define the `SkillCategory` Enum and the `AI_SKILLS` dictionary as specified.

**`SkillCategory` Enum:**

```python
class SkillCategory(str, Enum):
    ML_ENGINEERING = "ml_engineering"
    DATA_SCIENCE = "data_science"
    AI_INFRASTRUCTURE = "ai_infrastructure"
    AI_PRODUCT = "ai_product"
    AI_STRATEGY = "ai_strategy"
```

**`AI_SKILLS` Dictionary (truncated for brevity):**

```json
{
  "ml_engineering": [
    "pytorch",
    "tensorflow",
    "scikit-learn",
    "keras",
    "mlops",
    "model deployment",
    "gpu",
    "cuda",
    "deep learning",
    "reinforcement learning",
    "computer vision",
    "nlp",
    "natural language processing",
    "transformers",
    "generative ai",
    "llm",
    "large language models",
    "huggingface",
    "kubeflow",
    "sagemaker"
  ],
  "data_science": [
    "python",
    "r",
    "sql",
    "pandas",
    "numpy",
    "statistics",
    "experiment design",
    "ab testing",
    "predictive modeling",
    "data analysis",
    "feature engineering",
    "machine learning algorithms",
    "regression",
    "classification",
    "clustering",
    "time series analysis",
    "hypothesis testing"
  ],
  "ai_infrastructure": [
    "aws",
    "azure",
    "gcp",
    "kubernetes",
    "docker",
    "airflow",
    "spark",
    "hadoop",
    "kafka",
    "data bricks",
    "snowflake",
    "etl",
    "data pipelines",
    "big data",
    "cloud computing"
  ],
  "ai_product": [
    "product management",
    "ai product",
    "ux/ui",
    "market research",
    "customer journey",
    "go-to-market",
    "roadmap",
    "agile",
    "scrum",
    "user stories"
  ],
  "ai_strategy": [
    "ai strategy",
    "business intelligence",
    "roi",
    "stakeholder management",
    "digital transformation",
    "ethics",
    "governance",
    "compliance",
    "risk management"
  ]
}
```

The `AI_SKILLS` dictionary, with its structured categories, provides a clear framework for extracting specific AI competencies from job descriptions. This allows for a granular understanding of the demand for different types of AI expertise.

### 2.2 Define Seniority Levels and Classification Logic

To help HR understand the experience level of AI professionals being sought, you need to define job seniority levels and a function to classify them based on job titles. This function will be crucial for segmenting the talent market by experience. Remember to handle case-insensitivity for robustness.

**`SeniorityLevel` Enum:**

```python
class SeniorityLevel(str, Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    DIRECTOR = "director"
    VP = "vp"
    EXECUTIVE = "executive"
```

**`SENIORITY_INDICATORS` Dictionary:**

```json
{
  "entry": ["junior", "entry level", "associate", "intern"],
  "mid": ["mid level", "staff", "specialist"],
  "senior": ["senior", "sr.", "lead", "principal", "expert"],
  "lead": ["lead", "team lead", "manager"],
  "director": ["director"],
  "vp": ["vp", "vice president"],
  "executive": ["chief", "cto", "cio", "ceo", "cfo"]
}
```

The `classify_seniority` function (assumed in `source.py`) will use these indicators to assign a seniority level.

```python
def classify_seniority(title: str) -> SeniorityLevel:
    # Example implementation logic:
    title_lower = title.lower()
    for level, indicators in SENIORITY_INDICATORS.items():
        if any(ind in title_lower for ind in indicators):
            return SeniorityLevel(level)
    return SeniorityLevel.MID # Default if no indicator found
```

The `classify_seniority` function, by providing a standardized seniority level, allows the HR team to analyze demand across different experience brackets. This is vital for understanding talent supply and demand imbalances, and for tailoring recruitment efforts to specific experience profiles.

### 2.3 Implement AI Skill Extraction

Based on the `AI_SKILLS` taxonomy, you need a function to extract all relevant AI skills from a job description. This function will be applied to every job posting to build a comprehensive view of desired skills. Robustness here means ensuring case-insensitive matching.

```python
def extract_ai_skills(text: str) -> Set[str]:
    extracted = set()
    text_lower = text.lower()
    for category_skills in AI_SKILLS.values():
        for skill in category_skills:
            if skill.lower() in text_lower:
                extracted.add(skill)
    return extracted
```

The `extract_ai_skills` function is a cornerstone of this tool. By accurately identifying AI-related skills, HR can determine which specific competencies are most in demand, guiding training programs and recruitment focus. The case-insensitive matching prevents missing valuable data due to minor text variations.

### 3. Calculating the AI Relevance Score

To quantify how "AI-focused" each job role is, the HR team requires an **AI Relevance Score**. This score will help prioritize jobs, filter for highly specialized AI roles, and understand the depth of AI integration in various positions. Your task is to implement this scoring mechanism using the specified formula.

### 3.1 Implement AI Relevance Scoring Function

The AI relevance score combines the number of extracted AI skills with the explicit mention of AI-specific keywords in the job title. This provides a balanced view, considering both the depth of skill requirements and the role's explicit AI focus. The formula is designed to give more weight to the presence of skills while boosting for explicit AI terms in the title.

The formula for the AI Relevance Score (ranging from 0 to 1) is:
$$ \text{min}(\frac{\text{len(skills)}}{5}, 1.0) \times 0.6 + (0.4 \text{ if AI keywords present in title else } 0.0) $$
$$ \text{where } \text{len(skills)} \text{ is the number of unique AI skills extracted from the job description.} $$
$$ \text{The division by } 5 \text{ normalizes the skill count to a maximum of } 1.0 \text{ if } 5 \text{ or more unique skills are found.} $$
$$ \text{The } 0.6 \text{ and } 0.4 \text{ weights represent the contribution of skills and title keywords, respectively.} $$

AI keywords in title are `['ai', 'ml', 'machine learning', 'data scientist', 'mlops', 'artificial intelligence']`.

```python
def calculate_ai_relevance_score(skills: Set[str], title: str) -> float:
    score = 0.0
    # Skill component (60% weight, max 5 skills for 1.0 contribution)
    skill_component = min(len(skills) / 5.0, 1.0) * 0.6
    score += skill_component

    # Title component (40% weight if AI keywords are present)
    ai_title_keywords = ['ai', 'ml', 'machine learning', 'data scientist', 'mlops', 'artificial intelligence']
    title_lower = title.lower()
    if any(keyword in title_lower for keyword in ai_title_keywords):
        score += 0.4

    return min(score, 1.0) # Ensure score does not exceed 1.0
```

The AI Relevance Score function provides a quantifiable metric that HR can use to quickly identify and prioritize highly AI-centric roles. This helps them filter noise and focus recruitment efforts on positions that truly align with the company's AI strategy. A score closer to 1.0 indicates a deeply integrated AI role, while a lower score might suggest a more generalist role with some AI exposure.

### Test Logic on Sample Data

To verify the implemented logic, you can test it with a sample job title and description.

Suppose you have:
-   **Sample Job Title:** "Senior Machine Learning Engineer"
-   **Sample Job Description:** "We are looking for a deep learning engineer with experience in PyTorch, Transformers, and MLOps. Strong Python skills and familiar with AWS infrastructure."

Applying the functions:
```python
sample_title = "Senior Machine Learning Engineer"
sample_description = "We are looking for a deep learning engineer with experience in PyTorch, Transformers, and MLOps. Strong Python skills and familiar with AWS infrastructure."

extracted_skills = extract_ai_skills(sample_description)
seniority_level = classify_seniority(sample_title)
ai_relevance_score = calculate_ai_relevance_score(extracted_skills, sample_title)

print(f"Results for Sample Job Posting:")
print(f"- Extracted AI Skills: {extracted_skills}")
print(f"- Classified Seniority Level: {seniority_level.value.upper()}")
print(f"- AI Relevance Score: {ai_relevance_score:.2f}")
```

Expected output for the sample:
-   **Extracted AI Skills:** `{'python', 'deep learning', 'pytorch', 'mlops', 'transformers', 'aws'}` (order may vary)
-   **Classified Seniority Level:** `SENIOR`
-   **AI Relevance Score:** `1.00` (e.g., 6 skills = $min(6/5, 1.0) \times 0.6 = 1.0 \times 0.6 = 0.6$. "Machine Learning" in title contributes 0.4. Total = $0.6 + 0.4 = 1.0$)

## 3. Data Processing & Deduplication
Duration: 10:00

Now that all the individual logic components are defined, it's time to apply them to the entire dataset of raw job postings. A critical step in real-world data processing is **deduplication** to ensure data quality and prevent overcounting. You will process each job, extract features, and then clean the dataset.

### 3.1 Process Raw Job Postings

This step involves iterating through all generated job postings and applying the `extract_ai_skills`, `classify_seniority`, and `calculate_ai_relevance_score` functions. The results will be stored in a structured format suitable for further analysis.

The `process_job_postings` function (assumed in `source.py`) orchestrates this.

```python
# Assuming raw_job_postings has been generated from Step 1
if raw_job_postings is None:
    print("Please generate raw job postings first (from Step 1).")
else:
    processed_job_postings = process_job_postings(raw_job_postings)
    print(f"Processed {len(processed_job_postings)} job postings.")
    print("\nSample Processed Job Posting:")
    import json
    print(json.dumps(processed_job_postings[0], indent=2))
```

A sample of a processed job posting would look like this:

```json
{
  "title": "Lead ML Engineer",
  "description": "Seeking an experienced Lead ML Engineer to design, develop, and deploy machine learning models. Must have expertise in Python, TensorFlow, Kubernetes, and MLOps practices. Lead a team of data scientists and engineers.",
  "source": "LinkedIn",
  "url": "https://www.linkedin.com/jobs/123456789",
  "posted_date": "2023-01-15",
  "extracted_skills": ["python", "tensorflow", "kubernetes", "mlops", "machine learning"],
  "seniority_level": "lead",
  "ai_relevance_score": 1.0
}
```

Processing the raw job postings transforms unstructured text into structured features. This step is essential for converting raw data into a format that the HR team can analyze, providing a rich set of attributes for each job posting.

### 3.2 Deduplicate Processed Job Postings

Job postings from multiple sources can often be duplicates. To ensure accurate market analysis, it's crucial to identify and remove these redundant entries. The deduplication strategy involves creating a unique hash for each job based on a combination of its title, company, and location. This allows for robust identification of identical jobs.

The `deduplicate_job_postings` function (assumed in `source.py`) handles this.

```python
# Assuming processed_job_postings are available
if processed_job_postings is None:
    print("Please process job postings first (from Step 3.1).")
else:
    initial_count = len(processed_job_postings)
    deduplicated_list = deduplicate_job_postings(processed_job_postings)
    df_jobs = pd.DataFrame(deduplicated_list)
    df_jobs['posted_date'] = pd.to_datetime(df_jobs['posted_date'])

    print(f"Initial processed postings: {initial_count}")
    print(f"Deduplicated postings: {len(df_jobs)}")
    print(f"Removed {initial_count - len(df_jobs)} duplicates.")
    print("\nFirst 5 rows of the final DataFrame:")
    print(df_jobs.head())
```

A sample of the resulting `df_jobs` DataFrame:

| title                     | description                                             | source   | url                                 | posted_date | extracted_skills                                   | seniority_level | ai_relevance_score |
| : | : | :- | :- | :- | :- | :-- | :-- |
| Lead ML Engineer          | Seeking an experienced Lead ML Engineer...              | LinkedIn | https://www.linkedin.com/jobs/123... | 2023-01-15  | ['python', 'tensorflow', 'mlops']                  | lead            | 1.0                |
| Data Scientist            | Analyze large datasets, build predictive models...      | Indeed   | https://www.indeed.com/jobs/456...  | 2023-01-16  | ['python', 'pandas', 'statistics', 'sql']          | mid             | 0.8                |
| AI Product Manager        | Drive AI product strategy and roadmap...                | Glassdoor| https://www.glassdoor.com/jobs/789...| 2023-01-17  | ['ai product', 'product management']               | senior          | 0.6                |
| Junior Machine Learning Eng| Assist in developing ML models under supervision...     | LinkedIn | https://www.linkedin.com/jobs/101... | 2023-01-18  | ['python', 'scikit-learn']                         | entry           | 0.8                |
| Principal AI Architect    | Design scalable AI infrastructure on AWS...             | CareerBldr|https://www.careerbuilder.com/jobs/202...| 2023-01-19  | ['aws', 'kubernetes', 'cloud computing']           | senior          | 0.8                |

Deduplication is a vital data quality step. By eliminating duplicate entries, you ensure that the subsequent analysis accurately reflects the actual market demand, rather than being skewed by multiple listings of the same job. This gives HR a more reliable view of the talent landscape. The resulting `df_jobs` DataFrame is now clean and ready for in-depth analysis.

## 4. Temporal Aggregation & Visualizations
Duration: 20:00

The HR and strategy teams need to track how demand for AI talent evolves over time. Your next task is to aggregate the processed job data on a weekly basis. This will enable time-series analysis of job volumes, popular skills, and demand distribution across seniority levels, providing crucial insights into market dynamics. The final step is to visualize these insights.

### 4.1 Aggregate Job Postings by Week

To understand trends, we need to group job postings by the week they were posted. This involves calculating the start of the week for each `posted_date` and then counting the number of jobs for each week.

The `aggregate_weekly_job_volume` and `aggregate_weekly_skills_and_seniority` functions (assumed in `source.py`) will perform these aggregations.

```python
# Assuming df_jobs DataFrame is available
if df_jobs is None:
    print("Please process and deduplicate job postings first (from Step 3).")
else:
    weekly_job_volume_df = aggregate_weekly_job_volume(df_jobs)
    weekly_top_skills_df, weekly_seniority_df = \
        aggregate_weekly_skills_and_seniority(df_jobs)

    print("Weekly data aggregated successfully!")
    print("\nWeekly Job Volume (first 5 weeks):")
    print(weekly_job_volume_df.head())
    print("\nWeekly Top Skills (first 5 entries):")
    print(weekly_top_skills_df.head())
    print("\nWeekly Seniority Distribution (first 5 entries):")
    print(weekly_seniority_df.head())
```

Sample of `weekly_job_volume_df`:

| week_start | job_count |
| : | :-- |
| 2023-01-02 | 120       |
| 2023-01-09 | 135       |
| 2023-01-16 | 142       |
| 2023-01-23 | 110       |
| 2023-01-30 | 150       |

Sample of `weekly_top_skills_df`:

| week_start | skill       | count |
| : | :- | :- |
| 2023-01-02 | python      | 80    |
| 2023-01-02 | tensorflow  | 45    |
| 2023-01-02 | aws         | 30    |
| 2023-01-09 | python      | 90    |
| 2023-01-09 | pytorch     | 50    |

Sample of `weekly_seniority_df`:

| week_start | seniority_level | count |
| : | :-- | :- |
| 2023-01-02 | senior          | 50    |
| 2023-01-02 | mid             | 40    |
| 2023-01-02 | lead            | 20    |
| 2023-01-09 | senior          | 60    |
| 2023-01-09 | mid             | 45    |

Aggregating job volume by week provides HR with a clear trend line, showing whether demand for AI talent is increasing or decreasing. This information is invaluable for capacity planning in recruitment and for understanding the overall health of the AI talent market.

### 4.2 Identify Top Weekly Skills and Seniority Distribution

Beyond just job volume, HR needs to know which skills are most in demand and how seniority distributions are shifting. You will aggregate the most frequently mentioned skills and the distribution of seniority levels on a weekly basis.

These aggregations provide HR with granular insights. Tracking weekly top skills helps identify emerging technologies or shifting demand for specific competencies. Analyzing seniority distribution over time reveals if the market is favoring junior, mid-level, or senior talent, allowing HR to adjust their recruitment campaigns and compensation strategies accordingly.

### 5. Visualizing AI Talent Market Insights
Duration: 15:00

The final step in your workflow is to present these complex insights in a clear, actionable format for the HR and strategy teams. Visualizations make it easy to grasp trends and make data-driven decisions. You will generate charts for weekly job posting trends, top 10 overall requested skills, and the distribution of demand across seniority levels.

To generate these visualizations, assuming `df_jobs`, `weekly_job_volume_df`, `weekly_top_skills_df`, and `weekly_seniority_df` are available:

### 5.1 Visualize Weekly Trends in AI Job Postings

This visualization will show the overall volume of AI job postings over time, helping HR understand the general trajectory of the AI talent market.

```python
# Assuming plot_weekly_job_volume is defined in source.py
plot_weekly_job_volume(weekly_job_volume_df)
plt.title('Weekly Trends in AI Job Postings')
plt.xlabel('Week Start Date')
plt.ylabel('Number of Job Postings')
plt.show() # In a real notebook/Streamlit, this would display the plot.
```

This chart provides a quick and clear overview of market activity. A rising trend indicates a hot market, requiring faster recruitment, while a flat or declining trend might suggest opportunities for more strategic sourcing or a need to re-evaluate internal skill development.

### 5.2 Visualize Top 10 Most Requested AI Skills

Understanding which specific AI skills are most sought after is critical for talent development and acquisition strategies. This bar chart will highlight the overall top 10 skills across the entire collected period.

```python
# Assuming plot_top_skills is defined in source.py
plot_top_skills(df_jobs)
plt.title('Top 10 Most Requested AI Skills (Overall)')
plt.xlabel('Skill')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show() # In a real notebook/Streamlit, this would display the plot.
```

This visualization directly informs HR about which skills are currently commanding the most market attention. This intelligence can guide decisions on internal training programs, university partnerships, and targeted marketing for job advertisements.

### 5.3 Visualize AI Job Demand Distribution Across Seniority Levels

Analyzing the distribution of demand across different seniority levels reveals where the most significant talent gaps or competitive pressures might exist.

```python
# Assuming plot_seniority_distribution is defined in source.py
plot_seniority_distribution(df_jobs)
plt.title('AI Job Demand Distribution Across Seniority Levels')
plt.xlabel('Seniority Level')
plt.ylabel('Number of Job Postings')
plt.show() # In a real notebook/Streamlit, this would display the plot.
```

This chart helps HR understand whether the market is saturated with junior roles, desperately seeking senior leaders, or balanced across experience levels. This insight is critical for tailoring recruitment campaigns and understanding the competitive landscape for specific talent segments. For example, a high demand for "Lead" or "Director" level positions might signal an executive talent gap.

## Conclusion
Duration: 5:00

Congratulations! As a Software Developer, you have successfully built the foundational components of an internal AI Talent Demand Tracker. You've navigated through simulating data acquisition, implemented sophisticated text processing for skill extraction and seniority classification, developed a quantitative AI relevance score, performed crucial data cleaning through deduplication, and aggregated insights for temporal analysis. Finally, you've transformed complex data into actionable visualizations that directly address the strategic needs of your HR and strategy teams.

This tool provides your organization with a dynamic way to understand the evolving demand for AI talent, enabling proactive adaptation of hiring strategies and a focused approach to identifying emerging skill trends. Your work directly contributes to the company's ability to stay agile and competitive in the rapidly changing landscape of AI talent.
