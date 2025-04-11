import requests
import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import subprocess
import atexit
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM

# Load environment variables
load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")
# DatasetteAPI class
class DatasetteAPI:
    def __init__(self, base_url):
        self.base_url = base_url

    def execute_sql(self, database_name, sql_query):
        url = f"{self.base_url}/{database_name}.json"
        params = {"sql": sql_query}
        try:
            response = requests.get(url, params=params, timeout=30)  # Add timeout
            response.raise_for_status()  # Raise HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

# Function to get database schema

def get_schema_with_data(db_path):
    schema = {}
    sample_data = {}

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Fetch table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]

                # Fetch detailed schema information
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                # Extract column information: name, type, nullable, and default value
                schema[table_name] = [{
                    "name": column[1],
                    "data type": column[2],
                    "not_null": bool(column[3]),
                    "default_value": column[4]
                } for column in columns]

                # Fetch sample data for the table
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
                rows = cursor.fetchall()

                sample_data[table_name] = rows

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        
    return schema, sample_data

# Terminate Datasette process on shutdown
datasette_process = None

def terminate_datasette():
    global datasette_process
    if datasette_process:
        datasette_process.terminate()
        print("Datasette process terminated.")

atexit.register(terminate_datasette)


def full_dashboard(db_path):
    try:
        conn = sqlite3.connect(db_path)

        # Summary Statistics
        st.subheader("Summary Statistics")
        total_users = pd.read_sql("SELECT COUNT(*) AS total_users FROM mdl_user;", conn).iloc[0, 0]
        total_courses = pd.read_sql("SELECT COUNT(*) AS total_courses FROM mdl_course;", conn).iloc[0, 0]
        total_enrollments = pd.read_sql("SELECT COUNT(*) AS total_enrollments FROM mdl_user_enrolments;", conn).iloc[0, 0]
        total_forum_posts = pd.read_sql("SELECT COUNT(*) AS total_posts FROM mdl_forum_posts;", conn).iloc[0, 0]
        total_grades = pd.read_sql("SELECT COUNT(*) AS total_grades FROM mdl_grade_grades WHERE finalgrade IS NOT NULL;", conn).iloc[0, 0]

        st.write(f"**Total Users:** {total_users}")
        st.write(f"**Total Courses:** {total_courses}")
        st.write(f"**Total Enrollments:** {total_enrollments}")
        st.write(f"**Total Forum Posts:** {total_forum_posts}")
        st.write(f"**Total Grades:** {total_grades}")

        # Distribution of Grades
        st.subheader("Distribution of Grades")
        grades_query = "SELECT finalgrade FROM mdl_grade_grades WHERE finalgrade IS NOT NULL;"
        grades = pd.read_sql(grades_query, conn)
        if not grades.empty:
            plt.figure(figsize=(10, 6))
            plt.hist(grades['finalgrade'], bins=20, edgecolor='k', alpha=0.7)
            plt.title("Distribution of Final Grades")
            plt.xlabel("Grade")
            plt.ylabel("Frequency")
            plt.grid()
            st.pyplot(plt)

        # Average Grade per Course
        st.subheader("Average Grade per Course")
        avg_grade_query = """
        SELECT c.fullname AS course, AVG(g.finalgrade) AS average_grade
        FROM mdl_grade_grades g
        JOIN mdl_grade_items gi ON g.itemid = gi.id
        JOIN mdl_course c ON gi.courseid = c.id
        WHERE g.finalgrade IS NOT NULL
        GROUP BY c.fullname
        ORDER BY average_grade DESC;
        """
        course_grades = pd.read_sql(avg_grade_query, conn)
        if not course_grades.empty:
            plt.figure(figsize=(12, 6))
            plt.bar(course_grades['course'], course_grades['average_grade'], color='skyblue')
            plt.xticks(rotation=45, ha='right')
            plt.title("Average Grade per Course")
            plt.xlabel("Course")
            plt.ylabel("Average Grade")
            plt.tight_layout()
            st.pyplot(plt)

        # Course Enrollment Trends
        st.subheader("Course Enrollment Trends")
        enrollments_query = """
        SELECT c.fullname AS course, COUNT(e.userid) AS enrollment_count, e.timecreated
        FROM mdl_user_enrolments e
        JOIN mdl_course c ON e.enrolid = c.id
        GROUP BY c.fullname, e.timecreated
        ORDER BY e.timecreated;
        """
        enrollments = pd.read_sql(enrollments_query, conn)
        if not enrollments.empty:
            enrollments['timecreated'] = pd.to_datetime(enrollments['timecreated'], unit='s')
            plt.figure(figsize=(12, 6))
            for course in enrollments['course'].unique():
                course_data = enrollments[enrollments['course'] == course]
                plt.plot(course_data['timecreated'], course_data['enrollment_count'], label=course)
            plt.title("Course Enrollment Trends Over Time")
            plt.xlabel("Date")
            plt.ylabel("Enrollment Count")
            plt.legend()
            plt.grid()
            st.pyplot(plt)

        # User Activity Heatmap
        st.subheader("User Activity Heatmap")
        user_activity_query = """
        SELECT userid, courseid, timeaccess
        FROM mdl_user_lastaccess
        WHERE timeaccess IS NOT NULL;
        """
        user_activity = pd.read_sql(user_activity_query, conn)
        if not user_activity.empty:
            user_activity['timeaccess'] = pd.to_datetime(user_activity['timeaccess'], unit='s')
            user_activity['hour'] = user_activity['timeaccess'].dt.hour
            user_activity['day'] = user_activity['timeaccess'].dt.day_name()
            heatmap_data = user_activity.pivot_table(index='day', columns='hour', aggfunc='size', fill_value=0)
            plt.figure(figsize=(12, 6))
            sns.heatmap(heatmap_data, cmap='coolwarm', annot=False)
            plt.title("User Activity Heatmap (Day vs Hour)")
            plt.xlabel("Hour of Day")
            plt.ylabel("Day of Week")
            st.pyplot(plt)

        # Forum Posts Per Day
        st.subheader("Forum Posts Per Day")
        forum_posts_query = "SELECT created FROM mdl_forum_posts WHERE created IS NOT NULL;"
        forum_posts = pd.read_sql(forum_posts_query, conn)
        if not forum_posts.empty:
            forum_posts['created'] = pd.to_datetime(forum_posts['created'], unit='s')
            forum_posts['date'] = forum_posts['created'].dt.date
            posts_per_day = forum_posts.groupby('date').size()
            plt.figure(figsize=(12, 6))
            posts_per_day.plot(kind='line')
            plt.title("Forum Posts Per Day")
            plt.xlabel("Date")
            plt.ylabel("Number of Posts")
            plt.grid()
            st.pyplot(plt)

        # Grade Comparison Between Courses
        st.subheader("Grade Comparison Between Courses")
        grade_comparison_query = """
        SELECT g.finalgrade, c.fullname AS course
        FROM mdl_grade_grades g
        JOIN mdl_grade_items gi ON g.itemid = gi.id
        JOIN mdl_course c ON gi.courseid = c.id
        WHERE g.finalgrade IS NOT NULL;
        """
        grade_comparison = pd.read_sql(grade_comparison_query, conn)
        if not grade_comparison.empty:
            plt.figure(figsize=(40, 6))
            sns.boxplot(data=grade_comparison, x='course', y='finalgrade')
            plt.title("Grade Comparison Between Courses")
            plt.xlabel("Course")
            plt.ylabel("Final Grade")
            plt.grid()
            st.pyplot(plt)

        # Dynamic Table Viewer
        st.subheader("Explore Data")
        table_names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()
        selected_table = st.selectbox("Select a Table to View", table_names)
        if selected_table:
            table_data = pd.read_sql(f"SELECT * FROM {selected_table} LIMIT 100;", conn)
            st.write(f"Displaying {selected_table} data:")
            st.dataframe(table_data)

        conn.close()

    except Exception as e:
        st.error(f"An error occurred: {e}")
        
few_shot_examples = """
User: Who is the top performer in CADE
SQL Query: SELECT DISTINCT T1.firstname, T1.lastname FROM user AS T1 INNER JOIN course_completions AS T2 ON T1.id = T2.userid INNER JOIN course AS T3 ON T2.course = T3.id WHERE T3.shortname LIKE '%CADE%' ORDER BY T2.timecompleted DESC LIMIT 1

User: courses taken by fathima
SQL Query: SELECT DISTINCT T1.fullname FROM course AS T1 INNER JOIN course_completions AS T2 ON T1.id = T2.course INNER JOIN user AS T3 ON T2.userid = T3.id WHERE T3.firstname = 'Fathima'

User: courses taken by taskeen
SQL Query: SELECT DISTINCT c.fullname FROM course c JOIN user_enrolments ue ON c.id = ue.enrolid JOIN user u ON ue.userid = u.id WHERE u.firstname = 'Taskeen';

User: List all courses in which Nikita Shivakumar is enrolled
SQL Query: SELECT DISTINCT mc.fullname FROM user AS mu JOIN user_enrolments AS mue ON mu.id = mue.userid JOIN course AS mc ON mue.enrolid = mc.id WHERE mu.firstname = 'Nikita' AND mu.lastname = 'Shivakumar';

User: What is the average final grade for each courses,give course name?
SQL Query: SELECT AVG(gg.finalgrade), c.fullname FROM grade_grades AS gg JOIN grade_items AS gi ON gg.itemid = gi.id JOIN course AS c ON gi.courseid = c.id WHERE gi.itemtype = 'course' GROUP BY c.fullname;

User: which user have grades highest in course HCM.Provide user name
SQL Query: SELECT DISTINCT T1.firstname, T1.lastname FROM user AS T1 INNER JOIN grade_grades AS T2 ON T1.id = T2.userid INNER JOIN grade_items AS T3 ON T2.itemid = T3.id INNER JOIN course AS T4 ON T3.courseid = T4.id WHERE T4.shortname = 'HCM' ORDER BY T2.finalgrade DESC LIMIT 1

User: Who is the top performer in Maths.Provide name
SQL Query: SELECT DISTINCT T1.firstname, T1.lastname FROM user AS T1 INNER JOIN grade_grades AS T2 ON T1.id = T2.userid INNER JOIN grade_items AS T3 ON T2.itemid = T3.id WHERE T3.itemname LIKE '%Maths%' ORDER BY T2.finalgrade DESC LIMIT 1;

User: compare grades of two courses CADE And Maths
SQL Query: SELECT AVG(CASE WHEN c.shortname = 'CADE' THEN gg.finalgrade ELSE NULL END) AS avg_cade_grade, AVG(CASE WHEN c.shortname = 'Maths' THEN gg.finalgrade ELSE NULL END) AS avg_maths_grade FROM grade_grades AS gg JOIN grade_items AS gi ON gg.itemid = gi.id JOIN course AS c ON gi.courseid = c.id WHERE c.shortname IN ('CADE', 'Maths');

User: course taken by krupa
SQL Query: SELECT DISTINCT c.fullname FROM user AS u JOIN user_enrolments AS ue ON u.id = ue.userid JOIN course AS c ON ue.enrolid = c.id WHERE u.firstname = 'Krupa';

User: show user and the assignment name in which they are enrolled
SQL Query: SELECT DISTINCT T1.firstname, T1.lastname, T3.fullname FROM user AS T1 INNER JOIN assign_submission AS T2 ON T1.id = T2.userid INNER JOIN course AS T3 ON T2.assignment = T3.id
"""


relationship = """
-- USER DEVICES TABLE
CREATE TABLE user_devices (
    id INTEGER PRIMARY KEY,
    userid INTEGER NOT NULL,
    appid TEXT,
    name TEXT,
    model TEXT,
    platform TEXT,
    version TEXT,
    pushid TEXT,
    uuid TEXT,
    publickey TEXT,
    timecreated INTEGER,
    timemodified INTEGER,
    FOREIGN KEY (userid) REFERENCES user(id) ON DELETE CASCADE
);

-- USER ENROLLMENTS
CREATE TABLE user_enrolments (
    id INTEGER PRIMARY KEY,
    status INTEGER NOT NULL,
    enrolid INTEGER NOT NULL,
    userid INTEGER NOT NULL,
    timestart INTEGER,
    timeend INTEGER,
    modifierid INTEGER,show me question attempts


    timecreated INTEGER,
    timemodified INTEGER,
    FOREIGN KEY (userid) REFERENCES user(id) ON DELETE CASCADE
);

-- COURSE MODULES
CREATE TABLE course_modules (
    id INTEGER PRIMARY KEY,
    course INTEGER NOT NULL,
    module TEXT NOT NULL,
    section TEXT,
    added INTEGER,
    completion INTEGER,
    completiongradeitemnumber INTEGER,
    FOREIGN KEY (course) REFERENCES course(id) ON DELETE CASCADE
);

-- FORUMS
CREATE TABLE forum (
    id INTEGER PRIMARY KEY,
    course INTEGER NOT NULL,
    name TEXT NOT NULL,
    intro TEXT,
    type TEXT,
    timecreated INTEGER,
    timemodified INTEGER,
    FOREIGN KEY (course) REFERENCES course(id) ON DELETE CASCADE
);

-- FORUM DISCUSSIONS
CREATE TABLE forum_discussions (
    id INTEGER PRIMARY KEY,
    course INTEGER NOT NULL,
    forum INTEGER NOT NULL,
    name TEXT NOT NULL,
    firstpost INTEGER NOT NULL,
    userid INTEGER NOT NULL,
    groupid INTEGER,
    timecreated INTEGER,
    FOREIGN KEY (forum) REFERENCES forum(id) ON DELETE CASCADE,
    FOREIGN KEY (userid) REFERENCES user(id) ON DELETE CASCADE
);

-- FORUM POSTS
CREATE TABLE forum_posts (
    id INTEGER PRIMARY KEY,
    discussion INTEGER NOT NULL,
    userid INTEGER NOT NULL,
    created INTEGER,
    modified INTEGER,
    FOREIGN KEY (discussion) REFERENCES forum_discussions(id) ON DELETE CASCADE,
    FOREIGN KEY (userid) REFERENCES user(id) ON DELETE CASCADE
);

-- GRADES
CREATE TABLE grade_items (
    id INTEGER PRIMARY KEY,
    courseid INTEGER NOT NULL,
    itemname TEXT NOT NULL,
    itemtype TEXT NOT NULL,
    grademax REAL,
    grademin REAL,
    timecreated INTEGER,
    timemodified INTEGER,
    FOREIGN KEY (courseid) REFERENCES course(id) ON DELETE CASCADE
);

CREATE TABLE grade_grades (
    id INTEGER PRIMARY KEY,
    itemid INTEGER NOT NULL,
    userid INTEGER NOT NULL,
    finalgrade REAL,
    rawgrade REAL,
    timecreated INTEGER,
    timemodified INTEGER,
    FOREIGN KEY (itemid) REFERENCES grade_items(id) ON DELETE CASCADE,
    FOREIGN KEY (userid) REFERENCES user(id) ON DELETE CASCADE
);

-- WORKSHOPS
CREATE TABLE workshop (
    id INTEGER PRIMARY KEY,
    course INTEGER NOT NULL,
    name TEXT NOT NULL,
    intro TEXT,
    strategy TEXT,
    usepeerassessment INTEGER,
    grade REAL,
    submissionstart INTEGER,
    submissionend INTEGER,
    assessmentstart INTEGER,
    assessmentend INTEGER,
    FOREIGN KEY (course) REFERENCES course(id) ON DELETE CASCADE
);

CREATE TABLE workshop_submissions (
    id INTEGER PRIMARY KEY,
    workshopid INTEGER NOT NULL,
    authorid INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    contentformat TEXT,
    grade REAL,
    timegraded INTEGER,
    published INTEGER,
    FOREIGN KEY (workshopid) REFERENCES workshop(id) ON DELETE CASCADE,
    FOREIGN KEY (authorid) REFERENCES user(id) ON DELETE CASCADE
);

CREATE TABLE workshop_assessments (
    id INTEGER PRIMARY KEY,
    submissionid INTEGER NOT NULL,
    reviewerid INTEGER NOT NULL,
    timecreated INTEGER,
    timemodified INTEGER,
    grade REAL,
    gradinggrade REAL,
    feedbackauthor TEXT,
    FOREIGN KEY (submissionid) REFERENCES workshop_submissions(id) ON DELETE CASCADE,
    FOREIGN KEY (reviewerid) REFERENCES user(id) ON DELETE CASCADE
);

-- WIKIS
CREATE TABLE wiki (
    id INTEGER PRIMARY KEY,
    course INTEGER NOT NULL,
    name TEXT NOT NULL,
    intro TEXT,
    timecreated INTEGER,
    timemodified INTEGER,
    wikimode TEXT,
    defaultformat TEXT,
    FOREIGN KEY (course) REFERENCES course(id) ON DELETE CASCADE
);

CREATE TABLE wiki_pages (
    id INTEGER PRIMARY KEY,
    subwikiid INTEGER NOT NULL,
    title TEXT NOT NULL,
    cachedcontent TEXT,
    timecreated INTEGER,
    timemodified INTEGER,
    userid INTEGER,
    pageviews INTEGER,
    FOREIGN KEY (userid) REFERENCES user(id) ON DELETE CASCADE
);

-- ZOOM MEETINGS
CREATE TABLE zoom (
    id INTEGER PRIMARY KEY,
    course INTEGER NOT NULL,
    intro TEXT,
    meeting_id TEXT NOT NULL,
    join_url TEXT NOT NULL,
    start_time INTEGER,
    duration INTEGER,
    timezone TEXT,
    password TEXT,
    option_auto_recording INTEGER,
    FOREIGN KEY (course) REFERENCES course(id) ON DELETE CASCADE
);

CREATE TABLE zoom_meeting_participants (
    id INTEGER PRIMARY KEY,
    userid INTEGER NOT NULL,
    zoomuserid TEXT,
    uuid TEXT,
    join_time INTEGER,
    leave_time INTEGER,
    duration INTEGER,
    name TEXT,
    FOREIGN KEY (userid) REFERENCES user(id) ON DELETE CASCADE
);

CREATE TABLE zoom_meeting_recordings (
    id INTEGER PRIMARY KEY,
    zoomid INTEGER NOT NULL,
    meetinguuid TEXT,
    zoomrecordingid TEXT,
    name TEXT,
    externalurl TEXT,
    passcode TEXT,
    recordingstart INTEGER,
    timecreated INTEGER,
    timemodified INTEGER,
    FOREIGN KEY (zoomid) REFERENCES zoom(id) ON DELETE CASCADE
);

-- VPL (Virtual Programming Lab)
CREATE TABLE vpl (
    id INTEGER PRIMARY KEY,
    course INTEGER NOT NULL,
    name TEXT NOT NULL,
    shortdescription TEXT,
    intro TEXT,
    startdate INTEGER,
    duedate INTEGER,
    maxfiles INTEGER,
    maxfilesize INTEGER,
    password TEXT,
    grade INTEGER,
    evaluate INTEGER,
    automaticgrading INTEGER,
    FOREIGN KEY (course) REFERENCES course(id) ON DELETE CASCADE
);

CREATE TABLE vpl_submissions (
    id INTEGER PRIMARY KEY,
    vpl INTEGER NOT NULL,
    userid INTEGER NOT NULL,
    datesubmitted INTEGER,
    grader INTEGER,
    dategraded INTEGER,
    grade REAL,
    mailed INTEGER,
    FOREIGN KEY (vpl) REFERENCES vpl(id) ON DELETE CASCADE,
    FOREIGN KEY (userid) REFERENCES user(id) ON DELETE CASCADE
);

-- Table: role_assignments
CREATE TABLE role_assignments (
    id BIGINT PRIMARY KEY,
    userid BIGINT REFERENCES user(id),
    roleid BIGINT REFERENCES role(id),
    contextid BIGINT REFERENCES context(id),
    timemodified BIGINT,
    modifierid BIGINT REFERENCES user(id),
    component VARCHAR(100),
    itemid BIGINT,
    sortorder BIGINT,
);

-- Table: logstore_standard_log
CREATE TABLE logstore_standard_log (
    id BIGINT PRIMARY KEY,
    userid BIGINT REFERENCES user(id),
    courseid BIGINT REFERENCES course(id),
    contextid BIGINT REFERENCES context(id),
    action VARCHAR(255),
    target VARCHAR(255),
);

-- Table: event
CREATE TABLE event (
    id BIGINT PRIMARY KEY,
    courseid BIGINT REFERENCES course(id),
    userid BIGINT REFERENCES user(id),
    eventtype VARCHAR(20),
    timestart BIGINT,
    timeduration BIGINT,
);


-- Table: question_response_count
CREATE TABLE question_response_count (
    id BIGINT PRIMARY KEY,
    analysisid BIGINT REFERENCES question_response_analysis(id),
    try BIGINT,
    rcount BIGINT,
);

-- Table: question_attempts
CREATE TABLE question_attempts (
    id BIGINT PRIMARY KEY,
    questionusageid BIGINT,
    slot BIGINT,
    questionid BIGINT,
    behaviour VARCHAR(255),
    maxmark DECIMAL(10, 5),
    minfraction DECIMAL(10, 5),
    maxfraction DECIMAL(10, 5),
);

-- Table: quiz_attempts
CREATE TABLE quiz_attempts (
    id BIGINT PRIMARY KEY,
    quiz BIGINT,
    userid BIGINT REFERENCES user(id),
    attempt MEDIUMINT,
    timestart BIGINT,
    timefinish BIGINT,
    state VARCHAR(255),
    sumgrades DECIMAL(10, 5),

-- Table: user_lastaccess
CREATE TABLE user_lastaccess (
    id BIGINT PRIMARY KEY,
    userid BIGINT REFERENCES user(id),
    courseid BIGINT REFERENCES course(id),
    timeaccess BIGINT,

-- Table: question_response_count
CREATE TABLE question_response_count(
    id BIGINT PRIMARY KEY,
    analysisid BIGINT REFERENCES question_response_analysis(id),
    try BIGINT,
    rcount BIGINT,

-- Table: cohort_members
CREATE TABLE cohort_members (
    id BIGINT PRIMARY KEY,
    cohortid BIGINT REFERENCES cohort(id),
    userid BIGINT REFERENCES user(id),
);

-- Table: modules
CREATE TABLE modules (
  "id" INTEGER,
  "name" TEXT,
  "cron" INTEGER,
  "lastcron" INTEGER,
  "search" TEXT,
  "visible" INTEGER
)
"""

# Main Streamlit app
st.title("Moodle Database Application")

# Sidebar navigation
app_mode = st.sidebar.selectbox("Choose the app mode:", ["Chatbot", "Dashboard"])
ollama_api_url = "http://35.224.72.66:11434"
groq_key = os.getenv("GROQ_API_KEY")
# ollama = OllamaLLM(model="gemma2:27b", base_url=ollama_api_url,temperature=1.5)
# Chatbot section
ollama = OllamaLLM(model="gemma2:27b", base_url=ollama_api_url,temperature=1.5)

if app_mode == "Chatbot":
    st.header("Chatbot")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi! How can I assist you today?"}]

    db_file = "adminer_data_updated.sqlite"
    

    if "datasette_started" not in st.session_state:
        st.session_state["datasette_started"] = False
    if "db_path" not in st.session_state:
        st.session_state["db_path"] = None

    if db_file and not st.session_state["datasette_started"]:
        # db_path = f"uploaded_{db_file}"
        st.session_state["db_path"] = db_file
        # with open(db_path, "wb") as f:
        #     f.write(db_file.read())

        st.info("Starting Datasette...")
        datasette_process = subprocess.Popen([
            "datasette", "serve", st.session_state["db_path"], "--cors", "--port", "8001"
        ])

        time.sleep(5)
        st.success("Datasette started at http://127.0.0.1:8001/")
        st.session_state["datasette_started"] = True
    # Display chat history
    datasette_api = DatasetteAPI("http://127.0.0.1:8001")
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your query here..."):
        # Append user input to messages
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # Process user input
        gemini = GoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=api_key, temperature=0.8)
        if not api_key:
            error_message = "Please provide your Google API Key."
            st.session_state["messages"].append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.markdown(error_message)
        elif st.session_state["db_path"] is None:
            error_message = "Please upload a database file first."
            st.session_state["messages"].append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.markdown(error_message)
        else:
            try:

                schema, sample_data = get_schema_with_data(st.session_state["db_path"])
                # Retry mechanism
                retries = 3  # Number of retries
                attempt = 0
                while attempt < retries:

                    try:
                        prompt_template = f"""<s>[INST] 

                        You are a natural language to SQL query generator. Given the following information:
                        - **Database Schema:** "{schema}"
                        - **Relationships Schema:** "{relationship}"
                        - **Few-Shot Examples for Guidance:** "{few_shot_examples}"

                        Generate a SQL query based on the user's query: "{prompt}".

                        **Requirements for the Generated Query:**
                        1. **Key Parameter Identification:** Extract essential parameters such as `user ID`, `date`, or `course`. If missing, infer reasonable values (e.g., today's date for date-related queries or the current user's ID).
                        2. **Placeholder Replacement:** Replace any placeholders like `:userid` or `:date` with appropriate inferred values.
                        3. **Syntactic Correctness:** Ensure the SQL query follows the correct syntax and adheres to the given database schema for both simple and complex queries.
                        4. **Schema-Informed Assumptions:** Infer values based on schema patterns and sample data when not explicitly specified.
                        5. **Avoid Duplicates:** Use appropriate clauses such as `DISTINCT` to prevent duplicate results.
                        6. **Exact Table Names:** Use the exact table and column names as provided in the schema without any renaming.
                        7. **Data Type Validation:** Validate and format values according to schema types (e.g., integers, strings).
                        8. **Error Correction:** Correct any spelling errors or incorrect formatting in the query, ensuring alignment with the schema.
                        9. **Data Type Handling:** Handle value types correctly, including distinguishing between integers and strings per schema requirements.
                        10. **Enrollment Terminology:** Treat the term "enrolled" as indicating user participation or involvement in the specified activity.
                        11. **Case Sensitivity:** Handle table and column name case sensitivity appropriately based on the provided schema and sample data.
                        12. **Relationship Analysis:** Properly interpret foreign key relationships and references based on the relationship schema.
                        13. **Few-Shot Example Usage:** Use the few-shot examples only for analyzing potential user queries, but **do not** use them as direct answers in the query.

                        **Output Requirements:**  
                        Generate **only** the SQL query without any additional text or explanation.

                        [/INST]"""

                        # Generate SQL query using LLM

                        with st.spinner(f"Processing your query... (Attempt {attempt + 1}/{retries})"):
                            response = ollama.invoke(prompt_template)
                            sql_query = response.replace('```sql\n', '').replace('```', '')

                        # st.write(sql_query)
                        # Execute SQL query
                        query_result = datasette_api.execute_sql(st.session_state["db_path"].split('.')[0], sql_query)
                        # st.write(schema)

                        # Process query result
                        if isinstance(query_result, dict) and "rows" in query_result:
                            data = pd.DataFrame(query_result["rows"], columns=query_result["columns"])
                            if not data.empty:
                                prompt = f"""
                                Please convert the table data below into a well-structured, natural language paragraph that reads smoothly and naturally. Present the information clearly and cohesively without bullet points or lists.
                                Don't add any thing extra or explanation.
                                Data: {data}
                                """
                                final_answer = ollama.invoke(prompt)

                                st.session_state["messages"].append({"role": "assistant", "content": final_answer})
                                # Display in the current chat window
                                with st.chat_message("assistant"):
                                    st.write(final_answer)
                                break  # Exit the retry loop if data is found
                            else:
                                no_data_message = f"No data found on attempt {attempt + 1}."
                                st.session_state["messages"].append({"role": "assistant", "content": no_data_message})
                                with st.chat_message("assistant"):
                                    st.markdown(no_data_message)
                        else:
                            error_message = f"Error executing query"
                            st.session_state["messages"].append({"role": "assistant", "content": error_message})
                            with st.chat_message("assistant"):
                                st.markdown(error_message)
                            continue  # Exit loop if there's an execution error

                    except Exception as e:
                        exception_message = f"An error occurred during attempt {attempt + 1}: {e}"
                        st.session_state["messages"].append({"role": "assistant", "content": exception_message})
                        with st.chat_message("assistant"):
                            st.markdown(exception_message)

                    # Increment attempt count and retry
                    attempt += 1
                    time.sleep(2)  # Optional: Add a delay between retries

                # If all retries fail
                if attempt == retries:
                    final_message = "All retry attempts failed. Please refine your query or try again later."
                    st.session_state["messages"].append({"role": "assistant", "content": final_message})
                    with st.chat_message("assistant"):
                        st.markdown(final_message)

            except Exception as e:
                # Handle unexpected errors
                exception_message = f"An error occurred: {e}"
                st.session_state["messages"].append({"role": "assistant", "content": exception_message})
                with st.chat_message("assistant"):
                    st.markdown(exception_message)

# Dashboard section
elif app_mode == "Dashboard":
    st.title("Moodle Dashboard")

    # Upload database
    uploaded_file = 'database.sqlite'

    if uploaded_file:
        # db_path = f"uploaded_{uploaded_file.name}"
        # with open(db_path, "wb") as f:
        #     f.write(uploaded_file.read())
        full_dashboard(uploaded_file)
    else:
        st.warning("Please upload a SQLite database file to view the dashboard.")
