# QuLab: AI Talent Demand Tracker (PE-Org AIR System Lab)

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title

**QuLab: Job Signals & Talent Data - AI Talent Demand Tracker**

## 📝 Description

This Streamlit application, developed as part of a QuantUniversity Lab (QuLab), simulates the core components of an "AI Talent Demand Tracker" system. The primary goal is to empower HR and strategy teams with crucial insights into the evolving demand for AI talent in the broader market. It transforms raw, simulated job posting data into actionable intelligence, enabling proactive adaptation of hiring strategies, identification of in-demand skills, and a clear understanding of the competitive landscape for AI professionals.

The application guides users through a multi-stage process:
1.  **Simulated Data Generation**: Creating a diverse set of synthetic job postings.
2.  **Defining AI Taxonomy & Scoring Logic**: Establishing a comprehensive AI skill taxonomy, seniority classification, and an AI relevance scoring mechanism.
3.  **Data Processing & Deduplication**: Applying the defined logic to the dataset and cleaning it by removing duplicate job postings.
4.  **Temporal Aggregation & Visualizations**: Aggregating processed data weekly to analyze trends in job volume, top skills, and seniority distribution, presenting these insights through interactive visualizations.

This tool is vital for organizations to stay agile and competitive in a fast-changing talent market by providing data-driven insights into the AI talent landscape.

## ✨ Features

The application provides the following key functionalities:

*   **Synthetic Job Posting Generation**: Simulates the collection of a large dataset (2000+) of unique job postings with realistic titles, descriptions, sources, and dates.
*   **AI Skill Taxonomy Definition**: Defines a structured taxonomy of AI skills across categories like ML Engineering, Data Science, AI Infrastructure, AI Product, and AI Strategy.
*   **Seniority Classification**: Implements logic to classify job postings into distinct seniority levels (Entry, Mid, Senior, Lead, Director, VP, Executive) based on job titles.
*   **AI Skill Extraction**: Extracts relevant AI skills from job descriptions using the defined taxonomy, handling case-insensitivity.
*   **AI Relevance Scoring**: Calculates a quantitative "AI Relevance Score" for each job posting, indicating its focus on AI, by combining the number of extracted AI skills and explicit AI keywords in the job title.
*   **Data Deduplication**: Identifies and removes duplicate job postings using a robust hashing mechanism based on job title, company, and location.
*   **Temporal Aggregation**: Aggregates job data weekly to track trends in:
    *   Overall AI job posting volume.
    *   Most frequently requested AI skills.
    *   Distribution of demand across different seniority levels.
*   **Interactive Visualizations**: Generates dynamic charts to visualize:
    *   Weekly trends in AI job postings over time.
    *   Top 10 most requested AI skills across the entire dataset.
    *   Distribution of AI job demand across various seniority levels.
*   **Step-by-Step Navigation**: A clear sidebar navigation guides users through the lab project's workflow, ensuring a structured learning and operational experience.

## 🚀 Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the Repository (if applicable):**
    If this project is hosted on a Git repository, first clone it:
    ```bash
    git clone <repository_url>
    cd qu_lab_ai_talent_tracker
    ```
    *(Assuming your main Streamlit file is named `app.py` and helper functions are in `source.py`)*

2.  **Create a Virtual Environment (Recommended):**
    It's good practice to create a virtual environment to manage dependencies:
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    The application relies on several Python libraries. You can install them using `pip`:
    ```bash
    pip install streamlit pandas matplotlib seaborn numpy faker
    ```
    *(Alternatively, create a `requirements.txt` file with the above libraries and run `pip install -r requirements.txt`)*

## 💡 Usage

To run the Streamlit application:

1.  **Activate your virtual environment** (if not already active).
2.  **Navigate to the project directory** where your `app.py` (or main Streamlit file) is located.
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
4.  Your browser should automatically open a new tab displaying the application (usually at `http://localhost:8501`). If not, copy and paste the URL from your terminal.

### Navigating the Application

The application is designed as a guided lab project. Use the **sidebar navigation** to move through the different stages:

*   **Introduction & Data Generation**: Start here to understand the project and generate the synthetic job posting data.
*   **Defining AI Taxonomy & Scoring Logic**: Move to this page to see the core logic for skill extraction, seniority classification, and AI relevance scoring. You can also test this logic on sample data.
*   **Data Processing & Deduplication**: Proceed here to apply the defined logic to the entire dataset and deduplicate the job postings.
*   **Temporal Aggregation & Visualizations**: The final stage for aggregating data by week and generating insightful visualizations.

**Important Notes:**
*   You must click the "Generate Synthetic Job Postings" button on the first page to create the initial dataset.
*   Subsequent pages will require you to click processing/aggregation buttons to generate the necessary data for the next steps.
*   The application uses Streamlit's session state to persist data across page navigations, so your generated and processed data will be available as you move between sections.

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit application file
├── source.py               # Contains all helper functions, enums, and dictionaries
|                           # (e.g., generate_synthetic_job_postings, extract_ai_skills,
|                           #  classify_seniority, calculate_ai_relevance_score,
|                           #  process_job_postings, deduplicate_job_postings,
|                           #  aggregation & plotting functions, SkillCategory, AI_SKILLS, etc.)
└── README.md               # This README file
```
*(This structure assumes helper functions are abstracted into `source.py` as indicated by `from source import *` in `app.py`)*

## 🛠️ Technology Stack

*   **Framework**: [Streamlit](https://streamlit.io/) (for building interactive web applications)
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/) (for data structures and analysis)
*   **Numerical Operations**: [NumPy](https://numpy.org/) (for numerical computing)
*   **Data Visualization**:
    *   [Matplotlib](https://matplotlib.org/) (for creating static, animated, and interactive visualizations)
    *   [Seaborn](https://seaborn.pydata.org/) (for statistical data visualization based on Matplotlib)
*   **Synthetic Data Generation**: [Faker](https://faker.readthedocs.io/en/master/) (for generating fake data)
*   **Programming Language**: Python 3.8+

## 🤝 Contributing

This project is primarily a lab exercise. However, if you have suggestions for improvements or bug fixes:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file (if applicable) for details. For educational purposes of QuantUniversity labs.

## 📞 Contact

For questions or inquiries related to this QuLab project, please reach out via:

*   **QuantUniversity Website**: [www.quantuniversity.com](https://www.quantuniversity.com/)
*   **General Inquiries**: `info@quantuniversity.com`

---
*Developed as part of a QuantUniversity Lab project.*


## License

## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
