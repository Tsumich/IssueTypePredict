from contextlib import asynccontextmanager
from fastapi import Body, FastAPI, HTTPException, Request
from jira import get_issues_for_update, learning, predict_issue_type
import schedule
import time

def predict_and_update_issue_type():
    #Цикл по проектам
    model, label_encoder = learning('SCRUM')
    issues_for_update = get_issues_for_update('SCRUM')
    for issue in issues_for_update:
        print(issue)
        predict_issue_type(model, label_encoder, issue)

app = FastAPI()
@app.on_event("startup")
async def lifespan():
    predict_and_update_issue_type()
    schedule.every().day.at("01:00").do(predict_and_update_issue_type)



@app.post("/predict-issue-type")
def predict_type(data = Body()):
    summary = data["summary"]
    description = data["description"] 
    result = predict_issue_type(summary, description)
    return result
 
@app.get("/")
def hello():
    return "Hello!"
