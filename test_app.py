
import pandas as pd
from streamlit.testing.v1 import AppTest

# Assuming 'source.py' is available in the test environment with necessary functions.

def test_initial_page_load_and_session_state():
    """
    Tests that the app loads correctly on the 'Introduction & Data Generation' page
    and that initial session state variables are set as expected.
    """
    at = AppTest.from_file("app.py").run()
    assert at.session_state["current_page"] == 'Introduction & Data Generation'
    assert at.session_state["raw_job_postings"] is None
    assert at.session_state["processed_job_postings"] is None
    assert at.session_state["df_jobs"] is None
    assert at.session_state["weekly_job_volume_df"] is None
    assert at.session_state["weekly_top_skills_df"] is None
    assert at.session_state["weekly_seniority_df"] is None
    assert at.title[0].value == "QuLab: Job Signals & Talent Data"
    assert at.header[0].value == "1. Environment Setup and Data Generation"


def test_page_navigation():
    """
    Tests navigation between all pages using the sidebar radio buttons.
    """
    at = AppTest.from_file("app.py").run()

    # Navigate to "Defining AI Taxonomy & Scoring Logic"
    at.sidebar.radio[0].set_value("Defining AI Taxonomy & Scoring Logic").run()
    assert at.session_state["current_page"] == "Defining AI Taxonomy & Scoring Logic"
    assert at.header[0].value == "2. Defining AI Talent Taxonomy and Seniority Classification Logic"

    # Navigate to "Data Processing & Deduplication"
    at.sidebar.radio[0].set_value("Data Processing & Deduplication").run()
    assert at.session_state["current_page"] == "Data Processing & Deduplication"
    assert at.header[0].value == "4. Processing and Deduplicating Job Postings"

    # Navigate to "Temporal Aggregation & Visualizations"
    at.sidebar.radio[0].set_value("Temporal Aggregation & Visualizations").run()
    assert at.session_state["current_page"] == "Temporal Aggregation & Visualizations"
    assert at.header[0].value == "5. Temporal Aggregation for Trend Analysis"

    # Navigate back to "Introduction & Data Generation"
    at.sidebar.radio[0].set_value("Introduction & Data Generation").run()
    assert at.session_state["current_page"] == "Introduction & Data Generation"
    assert at.header[0].value == "1. Environment Setup and Data Generation"


def test_generate_synthetic_job_postings():
    """
    Tests the "Generate Synthetic Job Postings" button on the Introduction page.
    """
    at = AppTest.from_file("app.py").run()
    assert at.session_state["raw_job_postings"] is None

    # Click the generate button
    at.button[0].click().run()

    # Verify session state and displayed messages
    assert at.session_state["raw_job_postings"] is not None
    assert len(at.session_state["raw_job_postings"]) == 2000
    assert at.success[0].value == "Generated 2000 raw job postings."
    assert "Sample Raw Job Posting:" in at.markdown[4].value


def test_ai_logic_on_sample_data():
    """
    Tests the AI skill extraction, seniority classification, and relevance scoring
    on sample data in the 'Defining AI Taxonomy & Scoring Logic' page.
    """
    at = AppTest.from_file("app.py")
    at.session_state.current_page = "Defining AI Taxonomy & Scoring Logic"
    at.run()

    assert at.header[0].value == "2. Defining AI Talent Taxonomy and Seniority Classification Logic"

    # Set sample inputs
    at.text_input[0].set_value("Junior Data Scientist").run()
    at.text_area[0].set_value("Seeking an entry-level professional with Python, SQL, and basic machine learning skills. Experience with data analysis and visualization is a plus.").run()

    # Click the test logic button
    at.button[0].click().run()

    # Assert the results displayed
    assert "Extracted AI Skills: {'data analysis', 'sql', 'machine learning', 'python', 'data science'}" in at.markdown[18].value
    assert "Classified Seniority Level: ENTRY" in at.markdown[19].value
    # The score depends on the exact implementation in source.py,
    # let's assume a specific value for a junior data scientist.
    # For robust testing, you might mock source.calculate_ai_relevance_score if its output isn't perfectly deterministic without a full source.py.
    # Given the formula min(len(skills)/5, 1.0) * 0.6 + (0.4 if AI keywords present in title else 0.0)
    # Skills: 5, Title: "Junior Data Scientist" (contains "data scientist")
    # Score = min(5/5, 1.0) * 0.6 + 0.4 = 1.0 * 0.6 + 0.4 = 1.0
    assert "AI Relevance Score: 1.00" in at.markdown[20].value


def test_process_and_deduplicate_job_postings():
    """
    Tests the processing and deduplication steps on the 'Data Processing & Deduplication' page.
    This test chains interactions from generating raw jobs to deduplication.
    """
    at = AppTest.from_file("app.py").run()

    # 1. Generate Raw Job Postings
    at.button[0].click().run() # Click "Generate Synthetic Job Postings"
    assert at.session_state["raw_job_postings"] is not None

    # 2. Navigate to Data Processing page
    at.sidebar.radio[0].set_value("Data Processing & Deduplication").run()
    assert at.session_state["current_page"] == "Data Processing & Deduplication"
    assert at.header[0].value == "4. Processing and Deduplicating Job Postings"

    # 3. Process All Job Postings
    at.button[0].click().run() # Click "Process All Job Postings"
    assert at.session_state["processed_job_postings"] is not None
    assert at.success[0].value.startswith("Processed")
    assert "Sample Processed Job Posting:" in at.markdown[1].value

    # 4. Deduplicate Job Postings
    at.button[1].click().run() # Click "Deduplicate Job Postings"
    assert at.session_state["df_jobs"] is not None
    assert isinstance(at.session_state["df_jobs"], pd.DataFrame)
    assert not at.session_state["df_jobs"].empty
    assert at.success[2].value.startswith("Removed") # Check for the removed duplicates message
    assert at.dataframe[0].value is not None # Check if a dataframe is displayed


def test_temporal_aggregation_and_visualizations():
    """
    Tests the temporal aggregation and visualization generation on the
    'Temporal Aggregation & Visualizations' page.
    This test builds upon previous data generation and processing steps.
    """
    at = AppTest.from_file("app.py").run()

    # Simulate previous steps by populating session state
    at.button[0].click().run() # Generate raw jobs
    at.sidebar.radio[0].set_value("Data Processing & Deduplication").run()
    at.button[0].click().run() # Process jobs
    at.button[1].click().run() # Deduplicate jobs
    assert at.session_state["df_jobs"] is not None

    # Navigate to Temporal Aggregation & Visualizations page
    at.sidebar.radio[0].set_value("Temporal Aggregation & Visualizations").run()
    assert at.session_state["current_page"] == "Temporal Aggregation & Visualizations"
    assert at.header[0].value == "5. Temporal Aggregation for Trend Analysis"

    # 1. Aggregate Data Weekly
    at.button[0].click().run() # Click "Aggregate Data Weekly"
    assert at.session_state["weekly_job_volume_df"] is not None
    assert at.session_state["weekly_top_skills_df"] is not None
    assert at.session_state["weekly_seniority_df"] is not None
    assert isinstance(at.session_state["weekly_job_volume_df"], pd.DataFrame)
    assert at.success[0].value == "Weekly data aggregated successfully!"
    assert at.dataframe[0].value is not None # Weekly Job Volume
    assert at.dataframe[1].value is not None # Weekly Top Skills
    assert at.dataframe[2].value is not None # Weekly Seniority Distribution

    # 2. Generate Visualizations
    at.button[1].click().run() # Click "Generate Visualizations"

    # AppTest doesn't directly expose matplotlib figures.
    # We assert the presence of markdown text that accompanies the plots,
    # implying the plotting functions were called and outputs were generated.
    assert "This chart provides a quick and clear overview of market activity." in at.markdown[5].value
    assert "This visualization directly informs HR about which skills are currently commanding the most market attention." in at.markdown[7].value
    assert "This chart helps HR understand whether the market is saturated with junior roles, desperately seeking senior leaders, or balanced across experience levels." in at.markdown[9].value

    # Verify that plt.close() was called (or at least that the spinner completed).
    # Since we can't directly check the Matplotlib state without mocking,
    # successful execution without errors and presence of subsequent markdown
    # is the best we can do with AppTest's current capabilities for st.pyplot.
