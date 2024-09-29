import os
import pandas as pd
from pandasql import sqldf
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonAstREPLTool

def load_tables(dir="tables"):
    for file in os.listdir(dir):
        df = pd.read_csv(f"{dir}/{file}")
        globals()[file.split('.')[0]] = df
load_tables("tables")

@tool
def simulate_database_operation(sql: str):
    '''根据sql语句操作数据库中表'''
    print("调用sql工具: ", sql)
    load_tables("tables")
    return sqldf(sql)

tools =[
    PythonAstREPLTool(),
    DuckDuckGoSearchRun(),
    simulate_database_operation,
]