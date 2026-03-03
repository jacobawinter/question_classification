import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, cohen_kappa_score
import re


labelled = pd.read_excel('data/raw/questions_zm_label_sample_0-200.xlsx')
labelled = labelled[['question_id', 'label_spending_req','label_particular']]
full = pd.read_csv('data/raw/questions_zm_combined.csv', encoding = 'latin-1')

df = full.merge(labelled, on = 'question_id', how = 'left')
df = df.dropna(subset = ['label_spending_req'])


df['label'] = df['label_spending_req'].str.contains('spend', case = False, na = False)
df['label'] = df['label_particular']==1

df['text'] = df['question']

df['label'].value_counts()

### Train test Split ###
### Split and then upsample ###
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=1 - train_ratio, random_state=123)
# test is now 10% of the initial data set
# validation is now 15% of the initial data set
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=123) 

train_dataset = pd.DataFrame({'text':X_train, 'label':y_train})
val_dataset = pd.DataFrame({'text':X_val, 'label':y_val})
test_dataset = pd.DataFrame({'text':X_test, 'label':y_test})

#nltk stopwords saved here, bc downloading them onto Colab is annoying
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
             'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
               'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                   'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
                     'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                       'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o',
                         're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                           "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
                             "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
#lemmatizer = WordNetLemmatizer()

def preprocess(df):
    #remove short sentences
    #df['text'] = df['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    #remove stopwords
    
    #Add custom stopwords
    stopwords.extend(['-new_para-', "order", "mr", "hon", "minister", 'speaker', "government", 
                  "people", "hear","madam","question",'thank', 'committee',
                  'point','house','sir','zambia','ministry','member','laughter',
                  'country','one','can','us','interruptions','yes','chair',
                  'motion','raised','floor','second','please','words', 'zambians',
                  'mospagebreak', 'interrupt','shame', 'hammer','heckle', "govern",
                  "countri","zambian",'minist','ministri',
                  "hous","peopl", "becaus","rule","time","speak","continu","rais","pleas","veri"])
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
    #remove punctuation
    df['text'] = df['text'].str.replace('[^\w\s]','')
    #lemmatize
    #df['text'] = df['text'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
    #remove numbers
    df['text'] = df['text'].str.replace('\d+', '')
    #remove whitespace
    df['text'] = df['text'].str.replace('\s+', ' ')
    #remove leading and trailing whitespace
    df['text'] = df['text'].str.strip()
    #remove empty rows
    df = df[df['text'].notna()]
    #remove rows with only whitespace
    df = df[df['text'] != '']

    #remove hear hear
    return(df)

def fit_data(train, test):

    X_train = train['text']
    y_train = train['label']
    X_test = test['text']
    y_test = test['label']

    #Vectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)
    
    X_train = vectorizer.transform(X_train)
    X_test  = vectorizer.transform(X_test)

    #tfidf
    tfidfconverter = TfidfTransformer()
    tfidfconverter.fit(X_train)
    tfidfconverter.fit(X_test)
    #Model
    pred_table = pd.DataFrame(data = {'method': [], 'precision': [], 'recall': [], 'accuracy': [], 'f1': [], 'cohen_k':[]})
    models = [MultinomialNB(), 
              SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, tol=None),
              LogisticRegression(),#n_jobs=1, C=1e5),
              RandomForestClassifier(),#n_estimators=200, max_depth=3, random_state=0),
              SVC(),
              LinearSVC(),
              ]

    for model in models:
        row = fit_model(X_train, y_train, X_test, y_test, model)
        #add row to df
        pred_table.loc[len(pred_table)] = row

    return(pred_table)


def fit_model(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #SK learn average acccuracy and F1
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    accuracy  = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred)

    m_name = re.sub(r'\([^)]*\)', '', str(model))
    #print(pred_table)
    return([m_name, precision, recall, accuracy, f1, kappa])


#depending on number run different models
#get evaluation for group 1

# Load data

#Combine train and val
train = pd.concat([train_dataset, val_dataset], ignore_index = True)

#Run model
train = preprocess(train)
test = preprocess(test_dataset)
tab = (fit_data(train, test))

print(tab)