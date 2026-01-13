import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from source import * 
from io import BytesIO 

# --- Application Layout and Navigation ---
st.set_page_config(page_title="QuLab: Job Signals & Talent Data", layout="wide")

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
    "Introduction & Data Generation",
    "Defining AI Taxonomy & Scoring Logic",
    "Data Processing & Deduplication",
    "Temporal Aggregation & Visualizations"
]

# safely get index
try:
    current_index = page_options.index(st.session_state.current_page)
except ValueError:
    current_index = 0

page_selection = st.sidebar.radio(
    "Go to",
    page_options,
    index=current_index
)

# Update session state for page change
if page_selection != st.session_state.current_page:
    st.session_state.current_page = page_selection
    st.rerun()

st.title("QuLab: Job Signals & Talent Data")
st.divider()

# --- Page: Introduction & Data Generation ---
if st.session_state.current_page == 'Introduction & Data Generation':
    st.header("1. Environment Setup and Data Generation")
    st.markdown(f"Welcome to your next lab for building the \"PE-Org AIR System\"! Your mission is to empower the HR and strategy teams with crucial insights into the evolving demand for AI talent in the broader market. The goal is to build an internal tool that provides actionable intelligence, enabling proactive adaptation of hiring strategies, focus on in-demand skills, and a clear understanding of the competitive landscape for AI professionals. This tool is vital for the company to stay agile and competitive in a fast-changing talent market.")
    st.markdown(f"Today, you'll be developing the core components of this system. This involves simulating the data collection, implementing sophisticated logic for extracting AI-related skills and classifying job seniority, calculating a critical AI relevance score, and finally, aggregating this data for time-series analysis and visualization. Every step you take is designed to directly address a business need, transforming raw data into strategic insights for your organization.")

    st.subheader("Learning Objectives")
    st.markdown(f"By the end of this notebook, you will have implemented a system that:")
    st.markdown(f"* Collects (simulated) job posting data, handling real-world issues like deduplication.")
    st.markdown(f"* Applies a comprehensive AI skill taxonomy to extract relevant skills from job descriptions.")
    st.markdown(f"* Classifies job seniority levels based on job titles.")
    st.markdown(f"* Calculates an AI relevance score for each job posting using a predefined formula.")
    st.markdown(f"* Aggregates data weekly to enable time-series analysis of AI talent market trends.")
    st.markdown(f"* Visualizes key insights, such as weekly job volume, top skills, and seniority distribution.")

    st.subheader("1.1 Install Required Libraries")
    st.markdown(f"As a Software Developer, the first step is always to ensure your environment is ready and that you have data to work with. Since live scraping can be complex and rate-limited within a single notebook, we'll simulate the data acquisition phase by generating a diverse set of synthetic job postings that mirror real-world data. This allows us to focus on the processing and analysis logic, which is your primary task.")
    st.code("!pip install pandas faker matplotlib seaborn numpy")

    st.subheader("1.2 Import Dependencies")
    st.markdown(f"Next, import all the Python libraries and modules that will be used throughout the notebook. This makes sure all necessary functionalities are available from the start.")
    st.code("import pandas as pd\nimport numpy as np\nfrom faker import Faker\nfrom datetime import datetime, timedelta\nimport random\nfrom collections import defaultdict, Counter\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport hashlib\nfrom typing import List, Set, Dict, Tuple\nfrom enum import Enum\n\n# Configure plot styles for better readability\nplt.style.use('seaborn-v0_8-darkgrid')\nsns.set_palette('viridis')")

    st.subheader("1.3 Simulate Job Posting Data Collection")
    st.markdown(f"The HR team needs a robust dataset to analyze. Your task is to generate at least 1,500 unique job postings, each with a title, a full description, the source, a URL, and a realistic `posted_date`. This synthetic dataset will serve as the foundation for all subsequent analysis, simulating the output of a job scraper.")

    if st.button("Generate Synthetic Job Postings", key="generate_jobs_btn"):
        with st.spinner("Generating job postings..."):
            st.session_state.raw_job_postings = generate_synthetic_job_postings(num_postings=2000)
            st.success(f"Generated {len(st.session_state.raw_job_postings)} raw job postings.")
            st.markdown(f"\nSample Raw Job Posting:")
            st.json(st.session_state.raw_job_postings[0])

    if st.session_state.raw_job_postings:
        st.markdown(f"The output above confirms that our simulated data generation is working, providing a base list of job postings for further analysis. This is the raw material that your HR team needs you to process into actionable insights.")

# --- Page: Defining AI Taxonomy & Scoring Logic ---
elif st.session_state.current_page == 'Defining AI Taxonomy & Scoring Logic':
    st.header("2. Defining AI Talent Taxonomy and Seniority Classification Logic")
    st.markdown(f"The HR team needs to categorize job roles efficiently. Your next task is to implement the core logic for identifying AI-related skills and classifying job seniority. This is crucial for understanding the skill landscape and the experience levels being sought in the market. This stage directly uses the predefined AI skill taxonomy and seniority indicators from the HR requirements.")

    st.subheader("2.1 Define AI Skill Taxonomy")
    st.markdown(f"The AI skill taxonomy is a comprehensive dictionary mapping categories of AI to specific keywords. This structure helps in systematically identifying diverse AI competencies within job descriptions. You'll define the `SkillCategory` Enum and the `AI_SKILLS` dictionary as specified.")

    st.markdown(f"**`SkillCategory` Enum:**")
    st.code("class SkillCategory(str, Enum):\n    ML_ENGINEERING = \"ml_engineering\"\n    DATA_SCIENCE = \"data_science\"\n    AI_INFRASTRUCTURE = \"ai_infrastructure\"\n    AI_PRODUCT = \"ai_product\"\n    AI_STRATEGY = \"ai_strategy\"")
    st.markdown(f"**`AI_SKILLS` Dictionary (truncated for brevity):**")
    st.json({k.value: list(v)[:3] + ['...'] for k, v in AI_SKILLS.items()})
    st.markdown(f"The `AI_SKILLS` dictionary, with its structured categories, provides a clear framework for extracting specific AI competencies from job descriptions. This allows for a granular understanding of the demand for different types of AI expertise.")

    st.subheader("2.2 Define Seniority Levels and Classification Logic")
    st.markdown(f"To help HR understand the experience level of AI professionals being sought, you need to define job seniority levels and a function to classify them based on job titles. This function will be crucial for segmenting the talent market by experience. Remember to handle case-insensitivity for robustness.")
    st.markdown(f"**`SeniorityLevel` Enum:**")
    st.code("class SeniorityLevel(str, Enum):\n    ENTRY = \"entry\"\n    MID = \"mid\"\n    SENIOR = \"senior\"\n    LEAD = \"lead\"\n    DIRECTOR = \"director\"\n    VP = \"vp\"\n    EXECUTIVE = \"executive\"")
    st.markdown(f"**`SENIORITY_INDICATORS` Dictionary:**")
    st.json({k.value: v for k, v in SENIORITY_INDICATORS.items()})
    st.markdown(f"The `classify_seniority` function, by providing a standardized seniority level, allows the HR team to analyze demand across different experience brackets. This is vital for understanding talent supply and demand imbalances, and for tailoring recruitment efforts to specific experience profiles.")

    st.subheader("2.3 Implement AI Skill Extraction")
    st.markdown(f"Based on the `AI_SKILLS` taxonomy, you need a function to extract all relevant AI skills from a job description. This function will be applied to every job posting to build a comprehensive view of desired skills. Robustness here means ensuring case-insensitive matching.")
    st.code("def extract_ai_skills(text: str) -> Set[str]:\n    # ... implementation details ...")
    st.markdown(f"The `extract_ai_skills` function is a cornerstone of this tool. By accurately identifying AI-related skills, HR can determine which specific competencies are most in demand, guiding training programs and recruitment focus. The case-insensitive matching prevents missing valuable data due to minor text variations.")

    st.markdown("---")
    st.header("3. Calculating the AI Relevance Score")
    st.markdown(f"To quantify how \"AI-focused\" each job role is, the HR team requires an **AI Relevance Score**. This score will help prioritize jobs, filter for highly specialized AI roles, and understand the depth of AI integration in various positions. Your task is to implement this scoring mechanism using the specified formula.")

    st.subheader("3.1 Implement AI Relevance Scoring Function")
    st.markdown(f"The AI relevance score combines the number of extracted AI skills with the explicit mention of AI-specific keywords in the job title. This provides a balanced view, considering both the depth of skill requirements and the role's explicit AI focus. The formula is designed to give more weight to the presence of skills while boosting for explicit AI terms in the title.")

    st.markdown(f"The formula for the AI Relevance Score (ranging from 0 to 1) is:")
    st.markdown(r"$$ \text{min}(\frac{\text{len(skills)}}{5}, 1.0) \times 0.6 + (0.4 \text{ if AI keywords present in title else } 0.0) $$")
    st.markdown(r"$$ \text{where } \text{len(skills)} \text{ is the number of unique AI skills extracted from the job description.} $$")
    st.markdown(r"$$ \text{The division by } 5 \text{ normalizes the skill count to a maximum of } 1.0 \text{ if } 5 \text{ or more unique skills are found.} $$")
    st.markdown(r"$$ \text{The } 0.6 \text{ and } 0.4 \text{ weights represent the contribution of skills and title keywords, respectively.} $$")
    st.markdown(f"AI keywords in title are ```['ai', 'ml', 'machine learning', 'data scientist', 'mlops', 'artificial intelligence']```.")

    st.code("def calculate_ai_relevance_score(skills: Set[str], title: str) -> float:\n    # ... implementation details ...")

    st.markdown(f"The AI Relevance Score function provides a quantifiable metric that HR can use to quickly identify and prioritize highly AI-centric roles. This helps them filter noise and focus recruitment efforts on positions that truly align with the company's AI strategy. A score closer to 1.0 indicates a deeply integrated AI role, while a lower score might suggest a more generalist role with some AI exposure.")

    st.subheader("Test Logic on Sample Data")
    sample_title = st.text_input("Enter a sample job title:", "Senior Machine Learning Engineer", key="sample_title")
    sample_description = st.text_area("Enter a sample job description:", "We are looking for a deep learning engineer with experience in PyTorch, Transformers, and MLOps. Strong Python skills and familiar with AWS infrastructure.", height=150, key="sample_desc")

    if st.button("Test AI Logic on Sample", key="test_logic_btn"):
        with st.spinner("Applying logic..."):
            extracted_skills = extract_ai_skills(sample_description)
            seniority_level = classify_seniority(sample_title)
            ai_relevance_score = calculate_ai_relevance_score(extracted_skills, sample_title)

            st.markdown(f"**Results for Sample Job Posting:**")
            st.markdown(f"- **Extracted AI Skills:** `{extracted_skills}`")
            st.markdown(f"- **Classified Seniority Level:** `{seniority_level.value.upper()}`")
            st.markdown(f"- **AI Relevance Score:** `{ai_relevance_score:.2f}`")

# --- Page: Data Processing & Deduplication ---
elif st.session_state.current_page == 'Data Processing & Deduplication':
    st.header("4. Processing and Deduplicating Job Postings")
    st.markdown(f"Now that all the individual logic components are defined, it's time to apply them to the entire dataset of raw job postings. A critical step in real-world data processing is **deduplication** to ensure data quality and prevent overcounting. You will process each job, extract features, and then clean the dataset.")

    st.subheader("4.1 Process Raw Job Postings")
    st.markdown(f"This step involves iterating through all generated job postings and applying the `extract_ai_skills`, `classify_seniority`, and `calculate_ai_relevance_score` functions. The results will be stored in a structured format suitable for further analysis.")

    if st.session_state.raw_job_postings is None:
        st.warning("Please generate raw job postings on the 'Introduction & Data Generation' page first.")
    else:
        if st.button("Process All Job Postings", key="process_jobs_btn"):
            with st.spinner("Processing job postings..."):
                st.session_state.processed_job_postings = process_job_postings(st.session_state.raw_job_postings)
                st.success(f"Processed {len(st.session_state.processed_job_postings)} job postings.")
                st.markdown(f"\nSample Processed Job Posting:")
                st.json(st.session_state.processed_job_postings[0])

        if st.session_state.processed_job_postings:
            st.markdown(f"Processing the raw job postings transforms unstructured text into structured features. This step is essential for converting raw data into a format that the HR team can analyze, providing a rich set of attributes for each job posting.")

            st.subheader("4.2 Deduplicate Processed Job Postings")
            st.markdown(f"Job postings from multiple sources can often be duplicates. To ensure accurate market analysis, it's crucial to identify and remove these redundant entries. The deduplication strategy involves creating a unique hash for each job based on a combination of its title, company, and location. This allows for robust identification of identical jobs.")

            if st.button("Deduplicate Job Postings", key="deduplicate_jobs_btn"):
                with st.spinner("Deduplicating job postings..."):
                    initial_count = len(st.session_state.processed_job_postings)
                    deduplicated_list = deduplicate_job_postings(st.session_state.processed_job_postings)
                    st.session_state.df_jobs = pd.DataFrame(deduplicated_list)
                    st.session_state.df_jobs['posted_date'] = pd.to_datetime(st.session_state.df_jobs['posted_date'])

                    st.success(f"Initial processed postings: {initial_count}")
                    st.success(f"Deduplicated postings: {len(st.session_state.df_jobs)}")
                    st.success(f"Removed {initial_count - len(st.session_state.df_jobs)} duplicates.")
                    st.markdown(f"\nFirst 5 rows of the final DataFrame:")
                    st.dataframe(st.session_state.df_jobs.head())

            if st.session_state.df_jobs is not None:
                st.markdown(f"Deduplication is a vital data quality step. By eliminating duplicate entries, you ensure that the subsequent analysis accurately reflects the actual market demand, rather than being skewed by multiple listings of the same job. This gives HR a more reliable view of the talent landscape. The resulting `df_jobs` DataFrame is now clean and ready for in-depth analysis.")

# --- Page: Temporal Aggregation & Visualizations ---
elif st.session_state.current_page == 'Temporal Aggregation & Visualizations':
    st.header("5. Temporal Aggregation for Trend Analysis")
    st.markdown(f"The HR and strategy teams need to track how demand for AI talent evolves over time. Your next task is to aggregate the processed job data on a weekly basis. This will enable time-series analysis of job volumes, popular skills, and demand distribution across seniority levels, providing crucial insights into market dynamics.")

    if st.session_state.df_jobs is None:
        st.warning("Please process and deduplicate job postings on the 'Data Processing & Deduplication' page first.")
    else:
        st.subheader("5.1 Aggregate Job Postings by Week")
        st.markdown(f"To understand trends, we need to group job postings by the week they were posted. This involves calculating the start of the week for each `posted_date` and then counting the number of jobs for each week.")

        if st.button("Aggregate Data Weekly", key="aggregate_data_btn"):
            with st.spinner("Aggregating weekly data..."):
                st.session_state.weekly_job_volume_df = aggregate_weekly_job_volume(st.session_state.df_jobs)
                st.session_state.weekly_top_skills_df, st.session_state.weekly_seniority_df = \
                    aggregate_weekly_skills_and_seniority(st.session_state.df_jobs)

                st.success("Weekly data aggregated successfully!")
                st.markdown(f"Weekly Job Volume (first 5 weeks):")
                st.dataframe(st.session_state.weekly_job_volume_df.head())
                st.markdown(f"\nWeekly Top Skills (first 5 entries):")
                st.dataframe(st.session_state.weekly_top_skills_df.head())
                st.markdown(f"\nWeekly Seniority Distribution (first 5 entries):")
                st.dataframe(st.session_state.weekly_seniority_df.head())

        if st.session_state.weekly_job_volume_df is not None:
            st.markdown(f"Aggregating job volume by week provides HR with a clear trend line, showing whether demand for AI talent is increasing or decreasing. This information is invaluable for capacity planning in recruitment and for understanding the overall health of the AI talent market.")

            st.subheader("5.2 Identify Top Weekly Skills and Seniority Distribution")
            st.markdown(f"Beyond just job volume, HR needs to know which skills are most in demand and how seniority distributions are shifting. You will aggregate the most frequently mentioned skills and the distribution of seniority levels on a weekly basis.")
            st.markdown(f"These aggregations provide HR with granular insights. Tracking weekly top skills helps identify emerging technologies or shifting demand for specific competencies. Analyzing seniority distribution over time reveals if the market is favoring junior, mid-level, or senior talent, allowing HR to adjust their recruitment campaigns and compensation strategies accordingly.")

            st.markdown("---")
            st.header("6. Visualizing AI Talent Market Insights")
            st.markdown(f"The final step in your workflow is to present these complex insights in a clear, actionable format for the HR and strategy teams. Visualizations make it easy to grasp trends and make data-driven decisions. You will generate charts for weekly job posting trends, top 10 overall requested skills, and the distribution of demand across seniority levels.")

            if st.button("Generate Visualizations", key="generate_viz_btn"):
                st.subheader("6.1 Visualize Weekly Trends in AI Job Postings")
                st.markdown(f"This visualization will show the overall volume of AI job postings over time, helping HR understand the general trajectory of the AI talent market.")
                with st.spinner("Generating weekly job volume plot..."):
                    plot_weekly_job_volume(st.session_state.weekly_job_volume_df)
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

                st.subheader("6.3 Visualize AI Job Demand Distribution Across Seniority Levels")
                st.markdown(f"Analyzing the distribution of demand across different seniority levels reveals where the most significant talent gaps or competitive pressures might exist.")
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
