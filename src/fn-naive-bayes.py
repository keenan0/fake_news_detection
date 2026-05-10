import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

columns = [
    'id', 'label', 'statement', 'subject', 'speaker', 'job_title', 
    'state', 'party', 'barely_true', 'false_counts', 'half_true', 
    'mostly_true', 'pants_fire', 'context'
]

#data sets loading
train_df = pd.read_csv('train.tsv', sep='\t', header=None, names=columns)
valid_df = pd.read_csv('valid.tsv', sep='\t', header=None, names=columns)
test_df  = pd.read_csv('test.tsv', sep='\t', header=None, names=columns)

#curatam textele de valori NaN
train_df['statement'] = train_df['statement'].fillna('')
test_df['statement'] = test_df['statement'].fillna('')

X_train = train_df['statement']
y_train = train_df['label']
X_test = test_df['statement']
y_test = test_df['label']

#vectorizarea textului 
#folosesc tf-idf
#eliminam stop words
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#definire si antrenarea modelului naive-bayes 
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

#predictii si evaluarea modelului
y_pred = nb_model.predict(X_test_vec)

print("REZULTATE EVALUARE (naive bayes - 6 clase) \n")
print(f"Acuratete generala: {accuracy_score(y_test, y_pred):.4f}\n")
print("Raport detaliat (Precision, Recall, F1-Score):")
#generam toate metricile per clasa
print(classification_report(y_test, y_pred, zero_division=0))

#generam si afisam matricea de confuzie
cm = confusion_matrix(y_test, y_pred, labels=nb_model.classes_)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=nb_model.classes_, yticklabels=nb_model.classes_)
plt.title('Matricie de confuzie - naive bayes')
plt.xlabel('Predictie (ce spune modelul)')
plt.ylabel('Adevarul (eticheta reala)')
plt.show()


print("\n TOP 20 CUVINTE CELE MAI INFLUENTE PENTRU FIECARE CLASA")
feature_names = vectorizer.get_feature_names_out()


for i, class_label in enumerate(nb_model.classes_):
    #indecsii celor mai folosite 20 cuv
    top20_indices = np.argsort(nb_model.feature_log_prob_[i])[-20:]
    
    #ordine inversa ca sa afisam de la cel mai important la cel mai putin
    top20_indices = top20_indices[::-1]
    top20_words = [feature_names[j] for j in top20_indices]
    
    print(f"\nClasa [{class_label.upper()}]:")
    print(", ".join(top20_words))