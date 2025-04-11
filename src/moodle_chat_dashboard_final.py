import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time
import subprocess
import atexit
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI

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
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

# Function to get database schema
def get_schema_with_data_and_relations(db_path):
    schema = {}
    sample_data = {}
    relationships = {}
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            schema[table_name] = [column[1] for column in columns]
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
            rows = cursor.fetchall()
            sample_data[table_name] = rows
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            foreign_keys = cursor.fetchall()
            relationships[table_name] = []
            for fk in foreign_keys:
                relationships[table_name].append({
                    "from_column": fk[3],
                    "to_table": fk[2],
                    "to_column": fk[4]
                })
    return schema, sample_data, relationships

# Function to get all table names
def get_table_names(db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        return [table[0] for table in tables]

# Function to get table data
def get_table_data(db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

# Function to generate insights using AI
def generate_insights(data):
    prompt = f"""
    Analyze the following data and provide summarised insights:
    {data.to_string()}
    """
    gemini = GoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=gemini_api_key, temperature=0)
    response = gemini.invoke(prompt)
    return response
def terminate_existing_datasette():
    global datasette_process
    if datasette_process and datasette_process.poll() is None:
        datasette_process.terminate()
        datasette_process.wait()
        print("Existing Datasette process terminated.")
def full_dashboard(db_path):
    try:
        conn = sqlite3.connect(db_path)

        # Summary Statistics
        st.subheader("Summary Statistics")
        total_users = pd.read_sql("SELECT COUNT(*) AS total_users FROM user;", conn).iloc[0,0]
        total_courses = pd.read_sql("SELECT COUNT(*) AS total_courses FROM course;", conn).iloc[0,0]
        total_enrollments = pd.read_sql("SELECT COUNT(*) AS total_enrollments FROM user_enrolments;", conn).iloc[0,0]
        total_forum_posts = pd.read_sql("SELECT COUNT(*) AS total_posts FROM forum_posts;", conn).iloc[0,0]
        total_grades = pd.read_sql("SELECT COUNT(*) AS total_grades FROM grade_grades WHERE finalgrade IS NOT NULL;", conn).iloc[0,0]

        st.write(f"**Total Users:** {total_users}")
        st.write(f"**Total Courses:** {total_courses}")
        st.write(f"**Total Enrollments:** {total_enrollments}")
        st.write(f"**Total Forum Posts:** {total_forum_posts}")
        st.write(f"**Total Grades:** {total_grades}")

        # Distribution of Grades
        st.subheader("Distribution of Grades")
        grades_query = "SELECT finalgrade FROM grade_grades WHERE finalgrade IS NOT NULL;"
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
        FROM grade_grades g
        JOIN grade_items gi ON g.itemid = gi.id
        JOIN course c ON gi.courseid = c.id
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
        FROM user_enrolments e
        JOIN course c ON e.enrolid = c.id
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
        FROM user_lastaccess
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
        forum_posts_query = "SELECT created FROM forum_posts WHERE created IS NOT NULL;"
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
        FROM grade_grades g
        JOIN grade_items gi ON g.itemid = gi.id
        JOIN course c ON gi.courseid = c.id
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
# Terminate Datasette process on shutdown
datasette_process = None
def terminate_datasette():
    global datasette_process
    if datasette_process:
        datasette_process.terminate()
        print("Datasette process terminated.")
atexit.register(terminate_datasette)

# Streamlit app
gemini = GoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=gemini_api_key, temperature=0)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! How can I assist you today?"}]

# Upload database
db_file = os.getenv("DATABASE_FILE")

if "datasette_started" not in st.session_state:
    st.session_state["datasette_started"] = False
if "db_path" not in st.session_state:
    st.session_state["db_path"] = None

if db_file and not st.session_state["datasette_started"]:
    # db_path = f"uploaded_{db_file.name}"
    st.session_state["db_path"] = db_file
    # with open(db_path, "wb") as f:
    #     f.write(db_file.read())
    schema, sample_data, _ = get_schema_with_data_and_relations(st.session_state['db_path'])
    with st.spinner("Generating Schema"):
        schema_prompt = f"""Given the schema: "{schema}", carefully analyze all tables and their relationships. Generate the SQL statements that define the connected relations (foreign keys, primary keys, etc.) between every table in the schema. Ensure that every table from the schema is included, and that all relationships between tables are captured without omitting any table or relation. Provide the SQL statements, concise code format of SQL Part Only.
        
        Output Format:

        --primary keys
        ALTER TABLE booking ADD PRIMARY KEY (id);
        ALTER TABLE booking_commercials ADD PRIMARY KEY (id);

        --foreign keys
        ALTER TABLE booking_commercials ADD FOREIGN KEY (booking_id) REFERENCES booking (id);
        ALTER TABLE booking_commercials ADD FOREIGN KEY (sub_branch_id) REFERENCES booking (sub_branch_id);

        --References
        """
        response_text = gemini.invoke(schema_prompt).strip('</think>')
        if '</think>' in response_text:
            response_text = response_text.split('</think>')[1].strip()
        st.session_state['generated_schema'] = response_text
    st.info("Starting Datasette...")
    datasette_process = subprocess.Popen([
        "datasette", "serve", st.session_state['db_path'], "--cors", "--port", "8003"
    ])
    time.sleep(5)
    st.session_state["datasette_started"] = True

datasette_api = DatasetteAPI("http://127.0.0.1:8003")
option = st.sidebar.selectbox("Select Mode", ["Chatbot", "Dashboard"])

if option == "Chatbot":
    st.title("Moodle DB Query Application")
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == 'data':
                with st.expander("View Data"):
                    st.write(message["content"])
            else:
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your query here..."):
        # Append user input to messages
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # Process user input
        if not gemini_api_key:
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
                # Generate SQL query
                schema, sample_data,relations = get_schema_with_data_and_relations(st.session_state["db_path"])
                chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]])
                prompt_template = f"""<s>[INST]
                You are a natural language to SQL query generator. Given the following database schema and row data of each table: "{schema}", and relations between tables: "{st.session_state['generated_schema']}", determine whether the user's query requires a SQL query.

                If the query involves database-related operations such as fetching, updating, inserting, or deleting data, generate a SQL query based on these requirements:
                            
                1. Identify key parameters from the user's query, such as `user ID`, `date`, or `course`. If not explicitly mentioned, infer likely values (e.g., use today's date for date-related queries or the current user's ID for user-related queries).
                2. Replace any placeholders like `:userid` or `:date` with the inferred values based on context or assumptions.
                3. Ensure the SQL query is syntactically correct and follows the database schema, accommodating both simple and complex queries.
                4. Leverage common patterns in the schema and sample data to infer values when they are not clearly specified.
                5. Ensure the query does not return duplicate values by using appropriate SQL clauses like DISTINCT.
                6. Use the exact table names as provided in the schema without renaming them.
                7. Validate and format values correctly according to the schema's types (e.g., integers, strings).
                8. Correct any spelling mistakes or incorrect formatting in the query, ensuring alignment with the database schema.
                9. Ensure the correct handling of value types such as integers or strings according to the schema's requirements.
                10. Handle case sensitivity properly for table and column names based on the given sample data and schema.
                11. Use the table names and columns given in the schema based solely on the user's query.
                12. If explicit foreign keys are not provided, infer potential relationships between tables based on common naming conventions (e.g., primary key and column name similarities such as `id`, `table_name_id`, or `fk_`). Automatically generate JOIN statements where necessary, based on column names and possible relationships.
                13. If multiple tables are involved in the user's query, automatically include the necessary JOINs based on inferred relationships. The tables should be joined correctly according to the foreign key and primary key references or common naming patterns.

                If the user's query does not involve database-related operations and appears to be a normal conversation, respond naturally as a conversational AI without generating SQL queries.

                Chat History:
                {chat_history}

                Generate only the appropriate response—either a SQL query or a normal conversational response—without any additional text or explanation.[/INST]"""

                with st.spinner("Processing your query..."):
                    response = gemini.invoke(prompt_template)
                    sql_query = response.replace('```sql\n', '').replace('```', '')
                print(sql_query)
                # Execute SQL query
                query_result = datasette_api.execute_sql(st.session_state["db_path"].split('/')[1].split('.')[0],sql_query)

                if "SELECT" in sql_query:
                    if isinstance(query_result, dict) and "rows" in query_result:
                        data = pd.DataFrame(query_result["rows"], columns=query_result["columns"])
                        if not data.empty:
                            # Convert DataFrame to Markdown for chat persistence
                            markdown_data = data.to_markdown(index=False)

                            # Append to session state for persistent chat history
                            st.session_state["messages"].append({"role": "assistant", "content": "Fetched data displayed below."})
                            with st.expander("View Data"):
                                st.write(markdown_data)
                            st.session_state["messages"].append({"role": "data", "content": f"```\n{markdown_data}\n```"})
                            prompt1 = f"""
                            Please convert the table data below into a well-structured, natural language paragraph to answer the query. Format it nicely.
                            Don't add any extra explanation. Keep it objective.Generate Beautiful HTML Code In Proper Format Without "\n" With CSS For Visualization That Can Be Made On Data And It Should Align with data.

                            Query: {prompt}
                            Data: {data}
                            
                            Output Format:
                            [
                            Answer,HTML Code(Visualization)
                            ]
                            """
                            final_answer = gemini.invoke(prompt1)
                            st.session_state["messages"].append(
                                    {"role": "assistant", "content": final_answer}
                            )
                            with st.chat_message("assistant"):
                                st.markdown(final_answer)
                        else:
                            no_data_message = "No data found."
                            st.session_state["messages"].append({"role": "assistant", "content": no_data_message})
                            with st.chat_message("assistant"):
                                st.markdown(no_data_message)
                    else:
                        error_message = f"Error executing query: {query_result.get('error', 'Unknown error')}"
                        st.session_state["messages"].append({"role": "assistant", "content": error_message})
                        with st.chat_message("assistant"):
                            st.markdown(error_message)
                else:
                    st.session_state["messages"].append({"role": "assistant", "content": sql_query})
                    with st.chat_message("assistant"):
                        st.write(sql_query)

            except Exception as e:
                exception_message = f"An error occurred: {e}"
                st.session_state["messages"].append({"role": "assistant", "content": exception_message})
                with st.chat_message("assistant"):
                    st.markdown(exception_message)

elif option == "Dashboard":
    # Sidebar for data exploration and visualizations
    st.title("Moodle Dashboard")
    st.session_state['db_path'] = db_file

    if st.session_state['db_path']:
        full_dashboard(st.session_state["db_path"])
    else:
        st.warning("Please upload a SQLite database file to view the dashboard.")