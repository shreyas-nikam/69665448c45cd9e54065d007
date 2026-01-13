id: 69665448c45cd9e54065d007_user_guide
summary: Job Signals & Talent Data User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Navigating AI Talent Signals with Streamlit

## 1. Introduction & Data Generation
Duration: 00:05:00

Welcome to QuLab's "PE-Org AIR System" codelab! In today's rapidly evolving job market, understanding the demand for AI talent is crucial for any organization. This application is designed to empower Human Resources (HR) and strategy teams with actionable insights into the AI talent landscape. By simulating data collection, implementing sophisticated logic for skill extraction and seniority classification, calculating an AI relevance score, and providing temporal analysis, this tool helps companies stay agile and competitive.

Through this interactive guide, you will learn how to use the application to transform raw job posting data into strategic intelligence, enabling your organization to adapt hiring strategies, focus on in-demand skills, and gain a clear understanding of the competitive environment for AI professionals.

### Learning Objectives

By the end of this codelab, you will understand how the application:

*   **Simulates** the collection of diverse job posting data, addressing real-world challenges like redundancy.
*   **Applies** a comprehensive AI skill taxonomy to identify relevant skills from job descriptions.
*   **Classifies** job seniority levels based on job titles.
*   **Calculates** an AI relevance score for each job posting to quantify its AI focus.
*   **Aggregates** data weekly to enable time-series analysis of AI talent market trends.
*   **Visualizes** key insights, such as weekly job volume, top skills, and seniority distribution, for strategic decision-making.

### Simulating Job Posting Data Collection

The first step in any data analysis project is to acquire the data. Since live scraping can be complex and time-consuming, this application simulates the acquisition of job posting data. This allows us to focus on the subsequent processing and analysis, which is the core challenge for HR teams.

The application conceptually prepares the environment by ensuring necessary libraries are available and imported.

```console
!pip install pandas faker matplotlib seaborn numpy
```

```python
import pandas as pd
import numpy as np
from faker import Faker
# ... other imports
```

Your task here is to generate a realistic dataset of job postings. This synthetic dataset will mimic real-world data, providing the foundation for all subsequent analysis.

Click the **Generate Synthetic Job Postings** button.

<aside class="positive">
<b>What happens when you click 'Generate Synthetic Job Postings'?</b> The application creates a list of simulated job postings, each with a title, description, source, URL, and a realistic `posted_date`. This acts as your raw input, just as if it came from a live job board scraper.
</aside>

Once generated, you'll see a confirmation message and a sample of the raw job posting data in JSON format. This output confirms that our simulated data generation is working, providing a base list of job postings for further analysis. This is the raw material that your HR team needs you to process into actionable insights.

## 2. Defining AI Talent Taxonomy and Scoring Logic
Duration: 00:08:00

With the raw data in hand, the next critical step is to give it meaning. The HR team needs to categorize job roles efficiently and quantify their AI focus. This section of the application implements the core logic for identifying AI-related skills and classifying job seniority, directly addressing HR requirements.

### Defining AI Skill Taxonomy

To systematically identify AI competencies, a comprehensive AI skill taxonomy is used. This is essentially a structured dictionary that maps different categories of AI to specific keywords.

The application utilizes a `SkillCategory` enumeration and a detailed `AI_SKILLS` dictionary. The `SkillCategory` helps to group related skills (e.g., ML Engineering, Data Science).

```python
class SkillCategory(str, Enum):
    ML_ENGINEERING = "ml_engineering"
    DATA_SCIENCE = "data_science"
    AI_INFRASTRUCTURE = "ai_infrastructure"
    AI_PRODUCT = "ai_product"
    AI_STRATEGY = "ai_strategy"
```

The `AI_SKILLS` dictionary (partially shown below) provides a clear framework for extracting specific AI competencies from job descriptions, allowing for a granular understanding of the demand for different types of AI expertise.

```json
{
  "ml_engineering": [
    "machine learning",
    "mlops",
    "deep learning",
    "..."
  ],
  "data_science": [
    "data science",
    "statistical modeling",
    "predictive analytics",
    "..."
  ],
  "ai_infrastructure": [
    "cloud computing",
    "aws",
    "azure",
    "..."
  ],
  "ai_product": [
    "product management",
    "ai product",
    "roadmap",
    "..."
  ],
  "ai_strategy": [
    "ai strategy",
    "business intelligence",
    "digital transformation",
    "..."
  ]
}
```

### Defining Seniority Levels and Classification Logic

To help HR understand the experience level of AI professionals being sought, the application defines various job seniority levels and a function to classify them based on job titles. This segmentation of the talent market by experience is vital.

The `SeniorityLevel` enumeration categorizes experience, and `SENIORITY_INDICATORS` map keywords to these levels.

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

```json
{
  "entry": [
    "junior",
    "entry-level",
    "associate"
  ],
  "mid": [
    "mid-level",
    "experienced",
    "analyst"
  ],
  "senior": [
    "senior",
    "lead",
    "principal"
  ],
  "lead": [
    "lead",
    "team lead",
    "manager"
  ],
  "director": [
    "director",
    "head of"
  ],
  "vp": [
    "vp",
    "vice president"
  ],
  "executive": [
    "cto",
    "ceo",
    "chief"
  ]
}
```

The `classify_seniority` function, by providing a standardized seniority level, allows the HR team to analyze demand across different experience brackets. This is vital for understanding talent supply and demand imbalances and for tailoring recruitment efforts.

### Implementing AI Skill Extraction

Based on the `AI_SKILLS` taxonomy, the application includes a function to extract all relevant AI skills from a job description. This function is applied to every job posting to build a comprehensive view of desired skills, ensuring case-insensitive matching for robustness.

```python
def extract_ai_skills(text: str) -> Set[str]:
    # ... implementation details ...
```

<aside class="positive">
The <b>`extract_ai_skills`</b> function is a cornerstone of this tool. By accurately identifying AI-related skills, HR can determine which specific competencies are most in demand, guiding training programs and recruitment focus.
</aside>



### Calculating the AI Relevance Score

To quantify how "AI-focused" each job role is, the HR team requires an **AI Relevance Score**. This score helps prioritize jobs, filter for highly specialized AI roles, and understand the depth of AI integration in various positions.

The AI relevance score combines the number of extracted AI skills with the explicit mention of AI-specific keywords in the job title. This provides a balanced view, considering both the depth of skill requirements and the role's explicit AI focus. The formula is designed to give more weight to the presence of skills while boosting for explicit AI terms in the title.

The formula for the AI Relevance Score (ranging from 0 to 1) is:

$$ \text{min}(\frac{\text{len(skills)}}{5}, 1.0) \times 0.6 + (0.4 \text{ if AI keywords present in title else } 0.0) $$

$$ \text{where } \text{len(skills)} \text{ is the number of unique AI skills extracted from the job description.} $$
$$ \text{The division by } 5 \text{ normalizes the skill count to a maximum of } 1.0 \text{ if } 5 \text{ or more unique skills are found.} $$
$$ \text{The } 0.6 \text{ and } 0.4 \text{ weights represent the contribution of skills and title keywords, respectively.} $$

AI keywords in title are `['ai', 'ml', 'machine learning', 'data scientist', 'mlops', 'artificial intelligence']`.

```python
def calculate_ai_relevance_score(skills: Set[str], title: str) -> float:
    # ... implementation details ...
```

<aside class="positive">
The <b>AI Relevance Score</b> provides a quantifiable metric that HR can use to quickly identify and prioritize highly AI-centric roles. This helps them filter noise and focus recruitment efforts on positions that truly align with the company's AI strategy. A score closer to 1.0 indicates a deeply integrated AI role.
</aside>

### Test Logic on Sample Data

This section allows you to test the `extract_ai_skills`, `classify_seniority`, and `calculate_ai_relevance_score` functions on arbitrary job titles and descriptions.

1.  Enter a sample job title in the "Enter a sample job title:" text box.
2.  Enter a sample job description in the "Enter a sample job description:" text area.
3.  Click the **Test AI Logic on Sample** button.

You will see the extracted AI skills, the classified seniority level, and the calculated AI relevance score for your sample input. This helps confirm that the defined logic is working as expected.

## 3. Data Processing & Deduplication
Duration: 00:07:00

Now that all the individual logic components (skill extraction, seniority classification, AI relevance scoring) are defined, it's time to apply them to the entire dataset of raw job postings. A critical step in real-world data processing is **deduplication** to ensure data quality and prevent overcounting, which can significantly skew market insights.

### Processing Raw Job Postings

This step involves iterating through all the job postings you generated in the first step and applying the `extract_ai_skills`, `classify_seniority`, and `calculate_ai_relevance_score` functions to each one. The results are stored in a structured format, enhancing the data for further analysis.

<aside class="negative">
If you haven't generated raw job postings in the "Introduction & Data Generation" page, the application will remind you to do so before proceeding with this step.
</aside>

Click the **Process All Job Postings** button.

<aside class="positive">
Processing the raw job postings transforms unstructured text into structured features. This step is essential for converting raw data into a format that the HR team can analyze, providing a rich set of attributes for each job posting.
</aside>

After processing, you'll see a confirmation of the number of processed postings and a sample of a processed job posting, which now includes fields like `extracted_skills`, `seniority_level`, and `ai_relevance_score`.

### Deduplicating Processed Job Postings

Job postings from multiple sources often contain duplicates. To ensure accurate market analysis, it's crucial to identify and remove these redundant entries. The deduplication strategy involves creating a unique identifier (a hash) for each job based on a combination of its title, company, and location. This allows for robust identification of identical jobs.

Click the **Deduplicate Job Postings** button.

You will see a summary of how many postings were initially processed, how many remain after deduplication, and how many duplicates were removed. The first few rows of the final, deduplicated DataFrame will also be displayed.

<aside class="positive">
<b>Deduplication is a vital data quality step.</b> By eliminating duplicate entries, you ensure that the subsequent analysis accurately reflects the actual market demand, rather than being skewed by multiple listings of the same job. This gives HR a more reliable view of the talent landscape. The resulting DataFrame is now clean and ready for in-depth analysis.
</aside>

## 4. Temporal Aggregation & Visualizations
Duration: 00:10:00

The HR and strategy teams need to track how demand for AI talent evolves over time. Your final task is to aggregate the processed job data on a weekly basis, enabling time-series analysis of job volumes, popular skills, and demand distribution across seniority levels. This provides crucial insights into market dynamics.

<aside class="negative">
If you haven't processed and deduplicated job postings in the "Data Processing & Deduplication" page, the application will prompt you to do so first.
</aside>

### Aggregating Job Postings by Week

To understand trends, we need to group job postings by the week they were posted. This involves calculating the start of the week for each `posted_date` and then counting the number of jobs, identifying top skills, and analyzing seniority for each week.

Click the **Aggregate Data Weekly** button.

The application will then display the first 5 entries of the weekly job volume, weekly top skills, and weekly seniority distribution dataframes.

<aside class="positive">
Aggregating job volume by week provides HR with a clear trend line, showing whether demand for AI talent is increasing or decreasing. This information is invaluable for capacity planning in recruitment and for understanding the overall health of the AI talent market.
</aside>

### Identify Top Weekly Skills and Seniority Distribution

Beyond just job volume, HR needs to know which skills are most in demand and how seniority distributions are shifting. The aggregation process also identifies the most frequently mentioned skills and the distribution of seniority levels on a weekly basis.

<aside class="positive">
These aggregations provide HR with granular insights. Tracking weekly top skills helps identify emerging technologies or shifting demand for specific competencies. Analyzing seniority distribution over time reveals if the market is favoring junior, mid-level, or senior talent, allowing HR to adjust their recruitment campaigns and compensation strategies accordingly.
</aside>



### Visualizing AI Talent Market Insights

The final step is to present these complex insights in a clear, actionable format for the HR and strategy teams. Visualizations make it easy to grasp trends and make data-driven decisions. You will generate charts for weekly job posting trends, top 10 overall requested skills, and the distribution of demand across seniority levels.

Click the **Generate Visualizations** button.

#### Visualize Weekly Trends in AI Job Postings

This visualization shows the overall volume of AI job postings over time, helping HR understand the general trajectory of the AI talent market.

<aside class="positive">
This chart provides a quick and clear overview of market activity. A rising trend indicates a hot market, requiring faster recruitment, while a flat or declining trend might suggest opportunities for more strategic sourcing or a need to re-evaluate internal skill development.
</aside>

#### Visualize Top 10 Most Requested AI Skills

Understanding which specific AI skills are most sought after is critical for talent development and acquisition strategies. This bar chart highlights the overall top 10 skills across the entire collected period.

<aside class="positive">
This visualization directly informs HR about which skills are currently commanding the most market attention. This intelligence can guide decisions on internal training programs, university partnerships, and targeted marketing for job advertisements.
</aside>

#### Visualize AI Job Demand Distribution Across Seniority Levels

Analyzing the distribution of demand across different seniority levels reveals where the most significant talent gaps or competitive pressures might exist.

<aside class="positive">
This chart helps HR understand whether the market is saturated with junior roles, desperately seeking senior leaders, or balanced across experience levels. This insight is critical for tailoring recruitment campaigns and understanding the competitive landscape for specific talent segments. For example, a high demand for "Lead" or "Director" level positions might signal an executive talent gap.
</aside>



## Conclusion

Congratulations! You have successfully navigated the foundational components of an internal AI Talent Demand Tracker using this Streamlit application. You've understood how to simulate data acquisition, applied sophisticated text processing for skill extraction and seniority classification, grasped the concept of a quantitative AI relevance score, performed crucial data cleaning through deduplication, and explored aggregated insights for temporal analysis. Finally, you've seen how complex data is transformed into actionable visualizations that directly address the strategic needs of your HR and strategy teams.

This tool provides your organization with a dynamic way to understand the evolving demand for AI talent, enabling proactive adaptation of hiring strategies and a focused approach to identifying emerging skill trends. Your work directly contributes to the company's ability to stay agile and competitive in the rapidly changing landscape of AI talent.
