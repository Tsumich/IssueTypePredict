import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def predict_issue_type(summary, description):
    data = pd.read_csv("data.csv")
    data['text'] = data['Summary'] + " " + data['Description']

    # Разделение данных
    X = data['text']
    y = data['IssueType']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
    print(X_train)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    # Обучение модели
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Предсказание
    y_pred = model.predict(X_test_vec)
    print(f"Точность модели: {accuracy_score(y_test, y_pred)}")

    
    print(classification_report(y_test, y_pred))

    
    # Сохранение модели и векторзатора
    joblib.dump(model, "jira_issue_classifier.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    loaded_model = joblib.load("jira_issue_classifier.pkl")
    loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    text = summary + " " + description
    text_vec = loaded_vectorizer.transform([text])
    return loaded_model.predict(text_vec)[0]


#predicted_issue_type = predict_issue_type(summary, description)
#print(predicted_issue_type)


