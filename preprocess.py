import pyodbc
import json

def connection():
    s = 'localhost' #Your server name 
    d = 'chatagent' 
    u = 'chatadmin' #Your login
    p = '12345' #Your login password
    cstr = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+s+';DATABASE='+d+';UID='+u+';PWD='+ p
    conn = pyodbc.connect(cstr)
    return conn


def pre_process():
    
    conn = connection()
    cursor = conn.cursor()
    data = []
    cursor.execute("SELECT intent, category FROM intents_data GROUP BY intent, category ORDER BY intent")
    intents_data = cursor.fetchall()
    for elem in intents_data:
        cursor.execute("SELECT * FROM dbo.intents_data WHERE intent = ?", elem[0])
        patterns = []
        rows = cursor.fetchall()
        for row in rows:
            intent = row[4]
            cursor.execute("SELECT * FROM responses WHERE intent = ?", intent)
            response = cursor.fetchall()
            response_txts = []
            patterns.append(row[2])            
            for res in response:
                txt = res[1]
                response_txts.append(txt)
            
        data.append({"tag": elem[0], "category": elem[1], "patterns":patterns, "responses":response_txts})
    conn.close()
    
    return data
    

output_data = pre_process()
output_json = {"intents":output_data}
# Writing to sample.json
with open("sample.json", "w") as outfile:
    json.dump(output_json, outfile)
    
print("Contents Written to the file")