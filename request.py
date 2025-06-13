import requests
from requests.auth import HTTPBasicAuth
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

# Данные для аутентификации
def get_issues():
    url = os.getenv("JIRA_SEARCH")
    auth = HTTPBasicAuth(os.getenv("JIRA_LOGIN"), os.getenv("JIRA_KEY"))

    # Параметры запроса
    headers = {"Accept": "application/json"}
    query = {
        "jql": "",  # Пустой JQL вернёт все задачи
        "maxResults": 1000,  # Максимальное количество задач (по умолчанию 50)
        "startAt": 0  # Начальная позиция (для постраничного вывода)
    }
    response = requests.get(url, headers=headers, auth=auth, params=query)
    data = response.json()
    if response.status_code == 200:
        issues = data.get("issues", [])
        print(f"Найдено задач: {len(issues)}")
        with open('data.csv', 'w', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["Summary", "Description", "IssueType"])
            data = []
            for issue in issues:
                summary = issue['fields']['summary']
                description = issue['fields']['description']['content'][0]['content'][0]['text']
                issueType = issue['fields']['issuetype']['name']
                #texts.append(f"{summary} {description} {issueType}")
                data.append({
                    "text": f"{summary} {description}",  # Объединяем summary и description
                    "issue_type": issueType
                })
            texts = [item["text"] for item in data]
            labels = [item["issue_type"] for item in data]
            for issue in issues:
                summary = issue['fields']['summary']
                checkDescription = issue['fields']['description']
                description = issue['fields']['description']['content'][0]['content'][0]['text'] if checkDescription != None else " "
                issue_type = [issue['fields']['issuetype']['name']]
                writer.writerow([summary, description, issue_type[0]])
            return texts, labels
    else:
        print(f"Ошибка: {response.status_code}, {data}")

texts, labels = get_issues()
  # Пример URL для ProxyAPI


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
def predictIssue():
    # Векторизация текста задач из жиры
    embeddings = get_embeddings(texts)
    #Векторизация меток из жиры
    label_encoder = LabelEncoder()
    encoded_labels = get_embeddings(labels) #label_encoder.fit_transform(labels)
    encoded_labels2 = label_encoder.fit_transform(labels)

    print(encoded_labels2)
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(embeddings, encoded_labels2, test_size=0.2, train_size=0.8, random_state=42, stratify=encoded_labels2)
    
    # Обучение модели
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    new_embedding = get_embeddings(['Ошибка'])[0]

    predicted_label = model.predict([new_embedding])[0]
    predicted_issue_type = label_encoder.inverse_transform([predicted_label])[0]
    print(f"Предсказанный issueType: {predicted_issue_type}")