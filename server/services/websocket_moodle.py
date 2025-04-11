import asyncio
import os
import requests
import sys
import sqlite3
import websockets
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import pandas as pd
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
db_path = os.getenv("DATABASE_FILE")
# Initialize Gemini and Datasette API
gemini_api_key = os.getenv("GOOGLE_API_KEY")
gemini = GoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=gemini_api_key, temperature=0.5)
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
print("Starting Datasette...")
datasette_process = subprocess.Popen([
    "datasette", "serve", db_path, "--cors", "--port", "8003"
])
schema, sample_data,_ = get_schema_with_data_and_relations(db_path)
datasette_api = DatasetteAPI("http://127.0.0.1:8003")
print("DATAASETTE RUNNING")
schema_prompt = f"""Given the schema: "{schema}", and sample data :"{sample_data}" carefully analyze all tables and their relationships. Generate the SQL statements that define the connected relations (foreign keys, primary keys, etc.) between every table in the schema. Ensure that every table from the schema is included, and that all relationships between tables are captured without omitting any table or relation. Provide the SQL statements, concise code format of SQL Part Only.

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
relations = response_text

async def handle_websocket(websocket, path):
    async for message in websocket:
        try:
            # Parse the incoming message (user query)
            user_query = json.loads(message)["query"]
            print(user_query)
            # Get schema and sample data
            
            prompt_template = f"""<s>[INST]
            You are a natural language to SQL query generator. Given the following database schema and row data of each table: "{schema}", and relations between tables: "{relations}" and user_query:"{user_query}", determine whether the user's query requires a SQL query.

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

            Generate only the appropriate response—either a SQL query or a normal conversational response—without any additional text or explanation.[/INST]"""
            sql_query = gemini.invoke(prompt_template).replace('```sql\n', '').replace('```', '')
            print(sql_query)
            # Execute the SQL query using Datasette API
            query_result = datasette_api.execute_sql("updated_database", sql_query)
            print(query_result)
            if "SELECT" in sql_query:
                    if isinstance(query_result, dict) and "rows" in query_result:
                        data = pd.DataFrame(query_result["rows"], columns=query_result["columns"])
                        print(data)
                        if not data.empty:
                            # Convert DataFrame to Markdown for chat persistence
                            markdown_data = data.to_markdown(index=False)

                            # Append to session state for persistent chat history
                            prompt1 = f"""
                            Please convert the table data below into a well-structured, natural language paragraph to answer the query. Format it nicely.
                            Don't add any extra explanation. Keep it objective.Generate Beautiful HTML Code In Proper Format Without "\n" With CSS For Visualization That Can Be Made On Data And It Should Align with data.

                            Query: {user_query}
                            Data: {data}
                            
                            Output Format:
                            [
                            Answer,HTML Code(Visualization)
                            ]
                            """
                            final_answer = gemini.invoke(prompt1)
                            response = {
                                "status": "Success",
                                "message": final_answer
                            }

                        else:
                            response = {
                                "status": "error",
                                "message": "No data found or query execution failed."
                            }
            else:
                response = {
                    "status": "error",
                    "message": "No data found or query execution failed."
                }
            
            # Send the response back to the client
            await websocket.send(json.dumps(response))
        
        except Exception as e:
            # Handle errors
            error_response = {
                "status": "error",
                "message": f"An error occurred: {str(e)}"
            }
            await websocket.send(json.dumps(error_response))

# Start the WebSocket server
async def start_websocket_server():
    async with websockets.serve(handle_websocket, "localhost", 8765):
        await asyncio.Future()  # Run forever

# Run the WebSocket server
if __name__ == "__main__":
    asyncio.run(start_websocket_server())