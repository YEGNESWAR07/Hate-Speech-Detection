# Hate-Speech-Detection
 In this project, we aim to classify Twitter data into different categories using a  machine learning model. The dataset consists of tweets that are label into various  classes and the goal is to build a text classification model to predict the class of  given tweet. 
The dataset used in this project is a CSV file with following columns” 
 Unnamed: 0: An index column that is not necessary for the classification 
task. 
 count: A count or numerical value, which is not used in this analysis. 
 hate speech: A count or binary indicator of hate speech. 
 offensive language: A count or binary indicator of offensive language. 
 neither: A count or binary indicator for tweets that fall into neither of the 
above categories. 
 class: The target variable representing the category of the tweet. 
 tweet: The text of the tweet that will be used for classification. 


[Hate Speech Detection.pdf](https://github.com/user-attachments/files/16231602/Hate.Speech.Detection.pdf)


import pandas as pd 
import re 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from 
sklearn.metrics 
import 
confusion_matrix 
import seaborn as sns 
import matplotlib.pyplot as plt 
# Load the dataset 
data = pd.read_csv("twitter.csv") 
# Drop the unnecessary columns 
data 
accuracy_score, 
= data.drop(columns=['Unnamed: 
'offensive_language', 'neither']) 
# Print column names to verify 
print(data.columns)   
# Data Clean
def preprocess_text(text): 
# Remove special characters, URLs, and lowercase 
text = text.lower() 
text = ' '.join(filter(lambda x: x[0] != '@', text.split()))   
text = re.sub(r'http\S+', '', text)   
text = re.sub(r'[^a-zA-Z\s]', '', text)   
# Tokenize, remove stop words, and lemmatize 
stop_words = set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer() 
words = text.split() 
words = [lemmatizer.lemmatize(word) for word in words if word not in 
stop_words] 
return ' '.join(words) 
data['tweet'] = data['tweet'].apply(preprocess_text) 
# Define feature and target variables 
X = data['tweet'] 
y = data['class']  # 'class' is the target variable 
# Split data into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42) 
# Feature extraction using TF-IDF 
vectorizer = TfidfVectorizer() 
X_train = vectorizer.fit_transform(X_train) 
X_test = vectorizer.transform(X_test) 
# Train a logistic regression model with class weight adjustment for imbalance 
model = LogisticRegression(max_iter=1000, class_weight='balanced')  # Added 
class_weight='balanced' 
model.fit(X_train, y_train) 
# Make predictions 
y_pred = model.predict(X_test) 
# Evaluate the model 
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("Classification Report:\n", classification_report(y_test, y_pred)) 
cm = confusion_matrix(y_test, y_pred, labels=model.classes_) 
plt.figure(figsize=(5, 5)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, 
yticklabels=model.classes_) 
plt.xlabel('Predicted Labels') 
plt.ylabel('True Labels') 
plt.title(' Matrix Heatmap') 
plt.show()
