import os
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from sklearn.metrics import accuracy_score
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def get_all_issues_from_project(project_key):
    url = os.getenv("JIRA_URL")
    auth = HTTPBasicAuth(os.getenv("JIRA_LOGIN"), os.getenv("JIRA_KEY"))
    
    headers = {"Accept": "application/json"}
    query = {
        "jql": f"project = {project_key} AND issueType IS NOT EMPTY",
    }
    response = requests.get(url, headers=headers, auth=auth, params=query)
    data = response.json()
    if response.status_code == 200:
        issues = data.get("issues", [])
        print(f"Найдено задач: {len(issues)}")
        data = []
        for issue in issues:
            summary = issue['fields']['summary']
            description = issue['fields']['description']['content'][0]['content'][0]['text']
            issueType = issue['fields']['issuetype']['name']
            data.append({
                "text": f"{summary} {description}",
                "issue_type": issueType
            })
        texts = [item["text"] for item in data]
        labels = [item["issue_type"] for item in data]
        return texts, labels
    else:
        print(f"Ошибка: {response.status_code}, {data}")

def get_embeddings(texts, model="text-embedding-ada-002"):
    headers = {
        "Authorization": f"Bearer {os.getenv("API_KEY")}",
        "Content-Type": "application/json"
    }
    data = {
        "input": texts,
        "model": model
    }
    response = requests.post(os.getenv("PROXY_API_URL"), headers=headers, json=data)
    if response.status_code == 200:
        return np.array([item['embedding'] for item in response.json()['data']])
    else:
        raise Exception(f"Ошибка: {response.status_code}, {response.text}")

def get_issues_for_update(project_key):
    url = os.getenv("JIRA_URL")
    auth = HTTPBasicAuth(os.getenv("JIRA_LOGIN"), os.getenv("JIRA_KEY"))

    # Параметры запроса
    headers = {"Accept": "application/json"}
    query = {
        "jql": f"project = {project_key} AND created >= -1d",
    }
    response = requests.get(url, headers=headers, auth=auth, params=query)
    data = response.json()
    if response.status_code == 200:
        issues = data.get("issues", [])
        print(f"Найдено задач: {len(issues)}")
        data = []
        for issue in issues:
            summary = issue['fields']['summary']
            description = issue['fields']['description']['content'][0]['content'][0]['text']
            data.append(
                [f"{summary} {description}" ]
            )
        return data
    else:
        print(f"Ошибка: {response.status_code}, {data}")

def learning(project_key):
    # Получение содержимого задач из проекта
    texts, labels = get_all_issues_from_project(project_key)

    # Векторизация summary description
    embeddings = get_embeddings(texts)
    # Векторизация issueType
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Разделение данных 
    X_train, X_test, y_train, y_test = train_test_split(embeddings, encoded_labels, test_size=0.2, train_size=0.8, random_state=42, stratify=encoded_labels)
    
    # Обучение модели
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    return model, label_encoder

def predict_issue_type(model, label_encoder, issue):
    new_embedding = get_embeddings(issue)[0]
    predicted_label = model.predict([new_embedding])[0]
    predicted_issue_type = label_encoder.inverse_transform([predicted_label])[0]
    print(f"Предсказанный issueType: {predicted_issue_type}")
