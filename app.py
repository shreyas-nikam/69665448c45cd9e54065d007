import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from source import *
from io import BytesIO
import streamlit_mermaid as stmd


def get_highlighted_pipeline_diagram(step_number):
    """
    Generates a Mermaid diagram string with 'useMaxWidth' disabled 
    so it fills the container.
    """

    # 1. Define the Mermaid configuration directive
    # This JSON block tells Mermaid to NOT constrain the width (allows full-width)
    config = "%%{init: {'flowchart': {'useMaxWidth': false}}}%%"

    steps = [
        ("A", "Scrape"),
        ("B", "Extract Skills"),
        ("C", "Classify Roles"),
        ("D", "Calculate Score"),
        ("E", "Aggregate"),
        ("F", "Visualize")
    ]

    # 2. Start with config + flowchart definition
    mermaid_lines = [config, "flowchart LR"]

    # 3. Construct nodes
    node_definitions = [f'{node_id}["{label}"]' for node_id, label in steps]
    connection_string = " --> ".join(node_definitions)
    mermaid_lines.append(f"    {connection_string}")

    # 4. Highlight logic
    if 1 <= step_number <= len(steps):
        target_node_id = steps[step_number - 1][0]
        style_def = (
            f"    style {target_node_id} "
            "fill:#ff9900,stroke:#333,stroke-width:2px,color:white"
        )
        mermaid_lines.append(style_def)

    return "\n".join(mermaid_lines)


# --- Application Layout and Navigation ---
st.set_page_config(
    page_title="QuLab: Job Signals & Talent Data", layout="wide")

# --- Session State Initialization ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Introduction & Data Generation'
if 'raw_job_postings' not in st.session_state:
    st.session_state.raw_job_postings = None
if 'processed_job_postings' not in st.session_state:
    st.session_state.processed_job_postings = None
if 'df_jobs' not in st.session_state:
    st.session_state.df_jobs = None
if 'weekly_job_volume_df' not in st.session_state:
    st.session_state.weekly_job_volume_df = None
if 'weekly_top_skills_df' not in st.session_state:
    st.session_state.weekly_top_skills_df = None
if 'weekly_seniority_df' not in st.session_state:
    st.session_state.weekly_seniority_df = None

st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.sidebar.title("Navigation")

page_options = [
    "Introduction",
    "Data Generation & Scraping",
    "Defining AI Taxonomy & Scoring Logic",
    "Data Processing & Deduplication",
    "Temporal Aggregation & Visualizations"
]

# safely get index
try:
    current_index = page_options.index(st.session_state.current_page)
except ValueError:
    current_index = 0

page_selection = st.sidebar.selectbox(
    "Go to",
    page_options,
    index=current_index
)

# Update session state for page change
if page_selection != st.session_state.current_page:
    st.session_state.current_page = page_selection
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("🎯 Key Objectives")
st.sidebar.markdown("""
- **Remember**:  List AI-related job skills and role categories
- **Understand**: Explain how job postings indicate AI investment
- **Apply**: Implement job scraping and skill extraction
- **Analyze**: Calculate talent dimension score from job signals
""")

st.sidebar.divider()
st.sidebar.subheader("🛠️ Tools Introduced")
st.sidebar.markdown("""
- **python-jobspy**: For job data scraping
- **Playwright (Optional)**: Browser automation for web scraping
- **BeautifulSoup**: HTML parsing
- **Pandas**: Data manipulation and analysis
""")

st.title("QuLab: Job Signals & Talent Data")
st.divider()

# --- Page: Introduction & Data Generation ---
if st.session_state.current_page == 'Introduction':
    st.header("1. Environment Setup and Data Generation")
    st.markdown(f"Welcome to your next lab for building the \"PE-Org AIR System\"! Your mission is to empower the HR and strategy teams with crucial insights into the evolving demand for AI talent in the broader market. The goal is to build an internal tool that provides actionable intelligence, enabling proactive adaptation of hiring strategies, focus on in-demand skills, and a clear understanding of the competitive landscape for AI professionals. This tool is vital for the company to stay agile and competitive in a fast-changing talent market.")
    st.markdown(f"Today, you'll be developing the core components of this system. This involves collecting real-world job data through web scraping, implementing sophisticated logic for extracting AI-related skills and classifying job seniority, calculating a critical AI relevance score, and finally, aggregating this data for time-series analysis and visualization. Every step you take is designed to directly address a business need, transforming raw data into strategic insights for your organization.")

    st.subheader("Learning Objectives")
    st.markdown(
        f"""By the end of this notebook, you will have implemented a system that:
* Collects job posting data from LinkedIn, handling real-world issues like deduplication.
* Classifies job seniority levels based on job titles.
* Calculates an AI relevance score for each job posting using a predefined formula.
* Aggregates data weekly to enable time-series analysis of AI talent market trends.
* Visualizes key insights, such as weekly job volume, top skills, and seniority distribution.
""")

    stmd.st_mermaid(get_highlighted_pipeline_diagram(0), width='stretch')
    st.info("**Note:** The classification, extraction and processing steps in this lab are just for demo. In real-world scenarios, these would require more robust implementations and validations.")


elif st.session_state.current_page == 'Data Generation & Scraping':
    st.subheader("1.1 Job Posting Data Collection")
    st.markdown(f"The HR team needs a robust dataset to analyze. Using web scraping techniques with Playwright for browser automation, we can collect job postings from LinkedIn. Each job posting includes a title, description, company name, location, source, URL, and posted date.")
    stmd.st_mermaid(get_highlighted_pipeline_diagram(1), width='stretch')

    with st.expander("Web Scraper Implementation"):
        st.markdown(
            "Below is the implementation of the LinkedIn job scraper using Playwright:")
        st.code('''
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

    return _jobspy_df_to_jobs(df)''', language='python')
    search_query = st.text_input(
        "Search Query", value="AI Engineer")
    # num_postings = st.number_input(

    #     "Number of postings to scrape from LinkedIn", min_value=10, max_value=500, value=50)
    num_postings = 50

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Scrape Job Postings", key="generate_jobs_btn", use_container_width=True):
            with st.spinner("Scraping job postings..."):
                st.session_state.raw_job_postings = scrape_jobs_from_multiple_sources(
                    search_query=search_query,
                    sources=["linkedin"],
                    max_results_per_source=num_postings
                )
                if st.session_state.raw_job_postings:
                    st.success(
                        f"Scraped {len(st.session_state.raw_job_postings)} raw job postings.")
                else:
                    st.warning(
                        "No job postings were scraped. Please try again.")

    with col2:
        if st.button("Use Simulated Data Instead", key="use_simulated_btn", use_container_width=True):
            with st.spinner("Generating simulated job postings..."):
                st.session_state.raw_job_postings = generate_synthetic_job_postings(
                    num_postings=100
                )
                st.success(
                    f"Generated {len(st.session_state.raw_job_postings)} simulated job postings.")

    # Display the data table if available
    if st.session_state.raw_job_postings:
        st.markdown("### Job Postings")
        df_display = pd.DataFrame(st.session_state.raw_job_postings)
        st.dataframe(df_display, use_container_width=True)

        st.info("**Note:** Sometimes, you might get soft-blocks while scraping from LinkedIn. If you notice fewer results than expected, consider using the simulated data option to proceed with the lab exercises.")

    if st.session_state.raw_job_postings:
        st.markdown(f"The scraped data provides real-world job postings for analysis. This is the raw material that your HR team needs you to process into actionable insights.")


# --- Page: Defining AI Taxonomy & Scoring Logic ---
elif st.session_state.current_page == 'Defining AI Taxonomy & Scoring Logic':
    st.header("2. Defining AI Talent Taxonomy and Seniority Classification Logic")
    st.markdown(f"The HR team needs to categorize job roles efficiently. Your next task is to implement the core logic for identifying AI-related skills and classifying job seniority. This is crucial for understanding the skill landscape and the experience levels being sought in the market. This stage directly uses the predefined AI skill taxonomy and seniority indicators from the HR requirements.")

    st.subheader("2.1 Define AI Skill Taxonomy")
    st.markdown(f"The AI skill taxonomy is a comprehensive dictionary mapping categories of AI to specific keywords. This structure helps in systematically identifying diverse AI competencies within job descriptions. You'll define the `SkillCategory` Enum and the `AI_SKILLS` dictionary as specified.")

    with st.expander(f"**`SkillCategory` Enum**"):
        st.code("class SkillCategory(str, Enum):\n    ML_ENGINEERING = \"ml_engineering\"\n    DATA_SCIENCE = \"data_science\"\n    AI_INFRASTRUCTURE = \"ai_infrastructure\"\n    AI_PRODUCT = \"ai_product\"\n    AI_STRATEGY = \"ai_strategy\"")

    with st.expander(f"**`AI_SKILLS` Dictionary**"):
        st.code('''AI_SKILLS: Dict[SkillCategory, Set[str]] = {
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
}''', language='python')

    st.markdown(f"The `AI_SKILLS` dictionary, with its structured categories, provides a clear framework for extracting specific AI competencies from job descriptions. This allows for a granular understanding of the demand for different types of AI expertise.")

    st.subheader("2.2 Define Seniority Levels and Classification Logic")
    stmd.st_mermaid(get_highlighted_pipeline_diagram(2), width='stretch')
    st.markdown(f"To help HR understand the experience level of AI professionals being sought, you need to define job seniority levels and a function to classify them based on job titles. This function will be crucial for segmenting the talent market by experience. Remember to handle case-insensitivity for robustness.")
    with st.expander(f"**`SeniorityLevel` Enum**"):
        st.code("class SeniorityLevel(str, Enum):\n    ENTRY = \"entry\"\n    MID = \"mid\"\n    SENIOR = \"senior\"\n    LEAD = \"lead\"\n    DIRECTOR = \"director\"\n    VP = \"vp\"\n    EXECUTIVE = \"executive\"")
    with st.expander(f"**`SENIORITY_INDICATORS` Dictionary**"):
        st.code('''SENIORITY_INDICATORS: Dict[SeniorityLevel, List[str]] = {
    SeniorityLevel.ENTRY: ["junior", "entry", "associate", "intern", "graduate", "new grad"],
    SeniorityLevel.MID: ["mid", "intermediate", "ii", "2", "staff"],
    SeniorityLevel.SENIOR: ["senior", "sr", "iii", "3", "experienced"],
    SeniorityLevel.LEAD: ["lead", "principal", "iv", "4", "group"],
    SeniorityLevel.DIRECTOR: ["director", "head of", "mgr"],
    SeniorityLevel.VP: ["vp", "vice president"],
    SeniorityLevel.EXECUTIVE: ["chief", "cto", "cdo", "cao", "evp", "svp"]
}''', language='python')
    st.markdown(f"The `classify_seniority` function, by providing a standardized seniority level, allows the HR team to analyze demand across different experience brackets. This is vital for understanding talent supply and demand imbalances, and for tailoring recruitment efforts to specific experience profiles.")

    st.subheader("2.3 Implement AI Skill Extraction")
    stmd.st_mermaid(get_highlighted_pipeline_diagram(3), width='stretch')
    st.markdown(f"Based on the `AI_SKILLS` taxonomy, you need a function to extract all relevant AI skills from a job description. This function will be applied to every job posting to build a comprehensive view of desired skills. Robustness here means ensuring case-insensitive matching.")
    with st.expander(f"**AI Skill Extraction Function**"):
        st.code(
            '''
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

''')
    st.markdown(f"The `extract_ai_skills` function is a cornerstone of this tool. By accurately identifying AI-related skills, HR can determine which specific competencies are most in demand, guiding training programs and recruitment focus. The case-insensitive matching prevents missing valuable data due to minor text variations.")

    st.markdown("---")
    st.header("3. Calculating the AI Relevance Score")
    st.markdown(f"To quantify how \"AI-focused\" each job role is, the HR team requires an **AI Relevance Score**. This score will help prioritize jobs, filter for highly specialized AI roles, and understand the depth of AI integration in various positions. Your task is to implement this scoring mechanism using the specified formula.")
    stmd.st_mermaid(get_highlighted_pipeline_diagram(4), width='stretch')
    st.subheader("3.1 Implement AI Relevance Scoring Function")
    st.markdown(f"The AI relevance score combines the number of extracted AI skills with the explicit mention of AI-specific keywords in the job title. This provides a balanced view, considering both the depth of skill requirements and the role's explicit AI focus. The formula is designed to give more weight to the presence of skills while boosting for explicit AI terms in the title.")

    st.markdown(
        f"The formula for the AI Relevance Score (ranging from 0 to 1) is:")
    st.markdown(
        r"""$$
\text{min}(\frac{\text{len(skills)}}{5}, 1.0) \times 0.6 + (0.4 \text{ if AI keywords present in title else } 0.0) 
$$""")
    st.markdown(
        r"""$$\text{where } \text{len(skills)} \text{ is the number of unique AI skills extracted from the job description.} $$""")
    st.markdown(
        r"""$$\text{The division by } 5 \text{ normalizes the skill count to a maximum of } 1.0 \text{ if } 5 \text{ or more unique skills are found.} $$""")
    st.markdown(
        r"""$$\text{The } 0.6 \text{ and } 0.4 \text{ weights represent the contribution of skills and title keywords, respectively.} $$""")
    st.markdown(
        f"AI keywords in title are ```['ai', 'ml', 'machine learning', 'data scientist', 'mlops', 'artificial intelligence']```.")

    with st.expander(f"**AI Relevance Score Function**"):
        st.code('''def calculate_ai_relevance_score(skills: Set[str], title: str) -> float:
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
    return min(base_score + title_boost, 1.0)''', language='python')

    st.markdown(f"The AI Relevance Score function provides a quantifiable metric that HR can use to quickly identify and prioritize highly AI-centric roles. This helps them filter noise and focus recruitment efforts on positions that truly align with the company's AI strategy. A score closer to 1.0 indicates a deeply integrated AI role, while a lower score might suggest a more generalist role with some AI exposure.")

    st.subheader("Test Logic on Sample Data")
    sample_title = st.text_input(
        "Enter a sample job title:", "Senior Machine Learning Engineer", key="sample_title")
    sample_description = st.text_area("Enter a sample job description:",
                                      "We are looking for a deep learning engineer with experience in PyTorch, Transformers, and MLOps. Strong Python skills and familiar with AWS infrastructure.", height=150, key="sample_desc")

    if st.button("Test AI Logic on Sample", key="test_logic_btn"):
        with st.spinner("Applying logic..."):
            extracted_skills = extract_ai_skills(sample_description)
            seniority_level = classify_seniority(sample_title)
            ai_relevance_score = calculate_ai_relevance_score(
                extracted_skills, sample_title)

            st.markdown(f"**Results for Sample Job Posting:**")
            st.markdown(f"- **Extracted AI Skills:** `{extracted_skills}`")
            st.markdown(
                f"- **Classified Seniority Level:** `{seniority_level.value.upper()}`")
            st.markdown(
                f"- **AI Relevance Score:** `{ai_relevance_score:.2f}`")

# --- Page: Data Processing & Deduplication ---
elif st.session_state.current_page == 'Data Processing & Deduplication':
    st.header("4. Processing and Deduplicating Job Postings")
    st.markdown(f"Now that all the individual logic components are defined, it's time to apply them to the entire dataset of raw job postings. A critical step in real-world data processing is **deduplication** to ensure data quality and prevent overcounting. You will process each job, extract features, and then clean the dataset.")

    st.subheader("4.1 Process Raw Job Postings")
    st.markdown(f"This step involves iterating through all generated job postings and applying the `extract_ai_skills`, `classify_seniority`, and `calculate_ai_relevance_score` functions. The results will be stored in a structured format suitable for further analysis.")

    if st.session_state.raw_job_postings is None:
        st.warning(
            "Please generate raw job postings on the 'Introduction & Data Generation' page first.")
    else:
        if st.button("Process All Job Postings", key="process_jobs_btn"):
            with st.spinner("Processing job postings..."):
                st.session_state.processed_job_postings = process_job_postings(
                    st.session_state.raw_job_postings)
                st.success(
                    f"Processed {len(st.session_state.processed_job_postings)} job postings.")

                # Display processed jobs in a table
                st.markdown("### Processed Job Postings")
                df_processed = pd.DataFrame(
                    st.session_state.processed_job_postings)
                st.dataframe(df_processed, use_container_width=True)

        if st.session_state.processed_job_postings:
            st.markdown(f"Processing the raw job postings transforms unstructured text into structured features. This step is essential for converting raw data into a format that the HR team can analyze, providing a rich set of attributes for each job posting.")

            st.subheader("4.2 Deduplicate Processed Job Postings")
            st.markdown(f"Job postings from multiple sources can often be duplicates. To ensure accurate market analysis, it's crucial to identify and remove these redundant entries. The deduplication strategy involves creating a unique hash for each job based on a combination of its title, company, and location. This allows for robust identification of identical jobs.")

            if st.button("Deduplicate Job Postings", key="deduplicate_jobs_btn"):
                with st.spinner("Deduplicating job postings..."):
                    initial_count = len(
                        st.session_state.processed_job_postings)
                    deduplicated_list = deduplicate_job_postings(
                        st.session_state.processed_job_postings)
                    st.session_state.df_jobs = pd.DataFrame(deduplicated_list)
                    st.session_state.df_jobs['posted_date'] = pd.to_datetime(
                        st.session_state.df_jobs['posted_date'])

                    st.success(f"Initial processed postings: {initial_count}")
                    st.success(
                        f"Deduplicated postings: {len(st.session_state.df_jobs)}")
                    st.success(
                        f"Removed {initial_count - len(st.session_state.df_jobs)} duplicates.")
                    st.markdown(f"\nFirst 5 rows of the final DataFrame:")
                    st.dataframe(st.session_state.df_jobs.head())

            if st.session_state.df_jobs is not None:
                st.markdown(f"Deduplication is a vital data quality step. By eliminating duplicate entries, you ensure that the subsequent analysis accurately reflects the actual market demand, rather than being skewed by multiple listings of the same job. This gives HR a more reliable view of the talent landscape. The resulting `df_jobs` DataFrame is now clean and ready for in-depth analysis.")

# --- Page: Temporal Aggregation & Visualizations ---
elif st.session_state.current_page == 'Temporal Aggregation & Visualizations':
    st.header("5. Temporal Aggregation for Trend Analysis")
    st.markdown(f"The HR and strategy teams need to track how demand for AI talent evolves over time. Your next task is to aggregate the processed job data on a weekly basis. This will enable time-series analysis of job volumes, popular skills, and demand distribution across seniority levels, providing crucial insights into market dynamics.")

    if st.session_state.df_jobs is None:
        st.warning(
            "Please process and deduplicate job postings on the 'Data Processing & Deduplication' page first.")
    else:
        stmd.st_mermaid(get_highlighted_pipeline_diagram(5), width='stretch')
        st.subheader("5.1 Aggregate Job Postings by Week")
        st.markdown(f"To understand trends, we need to group job postings by the week they were posted. This involves calculating the start of the week for each `posted_date` and then counting the number of jobs for each week.")

        if st.button("Aggregate Data Weekly", key="aggregate_data_btn"):
            with st.spinner("Aggregating weekly data..."):
                st.session_state.weekly_job_volume_df = aggregate_weekly_job_volume(
                    st.session_state.df_jobs)
                st.session_state.weekly_top_skills_df, st.session_state.weekly_seniority_df = \
                    aggregate_weekly_skills_and_seniority(
                        st.session_state.df_jobs)

                st.success("Weekly data aggregated successfully!")
                st.markdown(f"Weekly Job Volume (first 5 weeks):")
                st.dataframe(st.session_state.weekly_job_volume_df.head())
                st.markdown(f"\nWeekly Top Skills (first 5 entries):")
                st.dataframe(st.session_state.weekly_top_skills_df.head())
                st.markdown(
                    f"\nWeekly Seniority Distribution (first 5 entries):")
                st.dataframe(st.session_state.weekly_seniority_df.head())

        if st.session_state.weekly_job_volume_df is not None:
            st.markdown(f"Aggregating job volume by week provides HR with a clear trend line, showing whether demand for AI talent is increasing or decreasing. This information is invaluable for capacity planning in recruitment and for understanding the overall health of the AI talent market.")

            st.subheader(
                "5.2 Identify Top Weekly Skills and Seniority Distribution")
            st.markdown(f"Beyond just job volume, HR needs to know which skills are most in demand and how seniority distributions are shifting. You will aggregate the most frequently mentioned skills and the distribution of seniority levels on a weekly basis.")
            st.markdown(f"These aggregations provide HR with granular insights. Tracking weekly top skills helps identify emerging technologies or shifting demand for specific competencies. Analyzing seniority distribution over time reveals if the market is favoring junior, mid-level, or senior talent, allowing HR to adjust their recruitment campaigns and compensation strategies accordingly.")

            st.markdown("---")
            st.header("6. Visualizing AI Talent Market Insights")
            stmd.st_mermaid(
                get_highlighted_pipeline_diagram(6), width='stretch')
            st.markdown(f"The final step in your workflow is to present these complex insights in a clear, actionable format for the HR and strategy teams. Visualizations make it easy to grasp trends and make data-driven decisions. You will generate charts for weekly job posting trends, top 10 overall requested skills, and the distribution of demand across seniority levels.")

            if st.button("Generate Visualizations", key="generate_viz_btn"):
                st.subheader("6.1 Visualize Weekly Trends in AI Job Postings")
                st.markdown(
                    f"This visualization will show the overall volume of AI job postings over time, helping HR understand the general trajectory of the AI talent market.")
                with st.spinner("Generating weekly job volume plot..."):
                    plot_weekly_job_volume(
                        st.session_state.weekly_job_volume_df)
                    st.pyplot(plt.gcf())
                    plt.close()
                    st.markdown(f"This chart provides a quick and clear overview of market activity. A rising trend indicates a hot market, requiring faster recruitment, while a flat or declining trend might suggest opportunities for more strategic sourcing or a need to re-evaluate internal skill development.")

                st.subheader("6.2 Visualize Top 10 Most Requested AI Skills")
                st.markdown(f"Understanding which specific AI skills are most sought after is critical for talent development and acquisition strategies. This bar chart will highlight the overall top 10 skills across the entire collected period.")
                with st.spinner("Generating top skills plot..."):
                    plot_top_skills(st.session_state.df_jobs)
                    st.pyplot(plt.gcf())
                    plt.close()
                    st.markdown(f"This visualization directly informs HR about which skills are currently commanding the most market attention. This intelligence can guide decisions on internal training programs, university partnerships, and targeted marketing for job advertisements.")

                st.subheader(
                    "6.3 Visualize AI Job Demand Distribution Across Seniority Levels")
                st.markdown(
                    f"Analyzing the distribution of demand across different seniority levels reveals where the most significant talent gaps or competitive pressures might exist.")
                with st.spinner("Generating seniority distribution plot..."):
                    plot_seniority_distribution(st.session_state.df_jobs)
                    st.pyplot(plt.gcf())
                    plt.close()
                    st.markdown(f"This chart helps HR understand whether the market is saturated with junior roles, desperately seeking senior leaders, or balanced across experience levels. This insight is critical for tailoring recruitment campaigns and understanding the competitive landscape for specific talent segments. For example, a high demand for \"Lead\" or \"Director\" level positions might signal an executive talent gap.")

    st.markdown("---")
    st.header("Conclusion")
    st.markdown(f"Congratulations! As a Software Developer, you have successfully built the foundational components of an internal AI Talent Demand Tracker. You've navigated through simulating data acquisition, implemented sophisticated text processing for skill extraction and seniority classification, developed a quantitative AI relevance score, performed crucial data cleaning through deduplication, and aggregated insights for temporal analysis. Finally, you've transformed complex data into actionable visualizations that directly address the strategic needs of your HR and strategy teams.")
    st.markdown(f"This tool provides your organization with a dynamic way to understand the evolving demand for AI talent, enabling proactive adaptation of hiring strategies and a focused approach to identifying emerging skill trends. Your work directly contributes to the company's ability to stay agile and competitive in the rapidly changing landscape of AI talent.")

# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
