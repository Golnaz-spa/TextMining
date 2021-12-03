import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
from nltk.stem import SnowballStemmer
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#show all columns
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',15)
"""
It is all message that students use it in their messages. We want to find that which one is spam which one is ham(not spam)
Goal -->  creating a classifier that says that message is spam or ham.
Methods that used in this project are: Lasso, Ridge, and ElasticNet.
We Compared the Accuracy of each model for each classes in order to see that which method has better performance

We also do Text Mining in this project. Text Mining has 2 levels.
1- making the text read (like removing un-useful characters, lowering all characters)
2- predicting the model(find the message is spam or ham)
"""
#encoding='latin-1'  ---> that read varuioe characters. UTf8 for Farsi language
text=pd.read_csv('E:/file/spam.csv',encoding='latin-1')

#change name of columns
text.columns=['Category','Message']# Put some names on the columns
#Count the number of words in each message
text['word_count'] = text['Message'].agg(lambda x: len(x.split(" ")))
#Count the number of characters in each message
text['char_count'] = text['Message'].agg(len)
#Change all the words to lowercase
text['Message']=text['Message'].agg(lambda x:x.lower())
#Compare the features obtained so far for ham and spam messages
df2=text.groupby('Category')
print(df2.agg('mean'))
print(df2.agg('std'))
print(text)

"""
Regular Expression (import re) is powerful package. It has some functions useful when uses text
"""

#we want to find sign in all rows (Each Message)
text['signs'] = text['Message'].agg(lambda x:re.findall('[^\w\s]',x))
#it counts the number of sign in each Message and save it in  a new coulmn name 'n_of_sign' coulmn
text['n_of_sign'] = text['Message'].agg(lambda x: len(re.findall('[^\w\s]', x)))
#it remove the sign of Message and change in with space. If we do not want change the Message column we can create a new column
text['Message'] = text['Message'].agg(lambda x:re.sub('[^\w\s]','',x))
print(text)

#Stopwords
import nltk
nltk.download("stopwords")
stop = stopwords.words('english')

# we add some words to stopwords. we do the same thing for removing unnecessary words
stop.append('u')#sometimes use u instaed of you
stop.append('ur')#sometimes use ur instaed of your
stop.append('2')#sometimes use 2 instaed of two
stop.append('4')#sometimes use 4 instaed of for
#find the number of stopwords in each message
text['n_of_stopwords'] = text['Message'].agg(lambda x: len([w for w in x.split() if w in stop]))


#Stemming all words
"""
when we want to data mining we want to find the root of a word. For example prediction, predicted, predicting all of them has a same root.
"""
stemmer = SnowballStemmer("english")
text['Message'] = text['Message'].agg(lambda x:(" ").join([stemmer.stem(w) for w in x.split()]))

#divide the Message to spam and ham. If the number of class high for example 40, not need to do it
spamtext=text.loc[text['Category']=='spam',:].loc[:,'Message']
hamtext=text.loc[text['Category']=='ham',:].loc[:,'Message']

#remoove stopwords
spam_no_stop =spamtext.agg(lambda x:' '.join([word for word in x.split() if word not in stop]))
hamw_no_stop =hamtext.agg(lambda x:' '.join([word for word in x.split() if word not in stop]))

#count each word in order to find theat each word how many times repeated
spam_word_counts=spam_no_stop.str.split(expand=True).stack().value_counts()
ham_word_counts=hamw_no_stop.str.split(expand=True).stack().value_counts()

#give words (index uses to give rhe rows)
spam_word_counts.index
ham_word_counts.index

#Select the words frequently used in all messages- Here, only get words that repaet more than 20 times. One method is get the higher counts of a words and *0.05 like 384*0.05 = 18.3
spamwords_usable=spam_word_counts[spam_word_counts>=20]
hamwords_usable=ham_word_counts[ham_word_counts>=20]

#Find the words that are similar in both of the spam and ham. Make them one list by using set().
s1=set(spamwords_usable.index)
s2=set(hamwords_usable.index)
Union=s1.union(set(s2))
Union=list(Union)

# create a matrxi that has 5572 rows and len of Union columns which all argumnets are zero
allfeatures=np.zeros((text.shape[0],len(Union)))

#Create a matrix that each argumnt is a Count of each word in all messages
for i in range(len(Union)):
  allfeatures[:,i]=text['Message'].agg(lambda x:x.split().count(Union[i]))

#Combining dataframe with contact()
Complete_data=pd.concat([text,pd.DataFrame(allfeatures, columns= Union)],axis=1)

#define X and y of dataframe
X=Complete_data.iloc[:,2:]
y=Complete_data['Category']
from sklearn.preprocessing import scale

X=scale(X)

#convert spam and ham to 0 and 1
enc=LabelEncoder()
enc.fit(y)
y = enc.transform(y)

#we put 5 to have short time for running. but in real problem we should do 1000 times . with high repeat we can get better accuracy
repeat=5

acc_lasso_ham=np.empty(repeat)
acc_lasso_spam=np.empty(repeat)
acc_ridge_ham=np.empty(repeat)
acc_ridge_spam=np.empty(repeat)
acc_elnet_ham=np.empty(repeat)
acc_elnet_spam=np.empty(repeat)

for i in range(repeat):
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #cross validation for C
    lassologreg = LogisticRegression(C=15, penalty="l1", solver="liblinear")
    ridgelogreg = LogisticRegression(C=15, penalty="l2", solver="liblinear")

    #cross validation for alpha, l1_ratio
    #l1_ratio is 1 --> lasso  ,   l1_ratio is 0 -->ridge   , l1_ratio is between 0< <1 --> elasticnet
    elaslogreg = SGDClassifier(loss='log', penalty='elasticnet', alpha=0.0001, l1_ratio=0.5)

    lassologreg.fit(X_train, y_train)
    ridgelogreg.fit(X_train, y_train)
    elaslogreg.fit(X_train, y_train)

    #get Confusion Matrix of lasso, ridge, and elastic Net
    lasso = confusion_matrix(y_test, lassologreg.predict(X_test))
    ridge = confusion_matrix(y_test, ridgelogreg.predict(X_test))
    elnet = confusion_matrix(y_test, elaslogreg.predict(X_test))

    #Accuracy for class ham and spam in lasso, ridge, and elastic net
    acc_lasso_ham[i] = lasso[0, 0] / sum(lasso[0, :])
    acc_lasso_spam[i] = lasso[1, 1] / sum(lasso[1, :])
    acc_ridge_ham[i] = ridge[0, 0] / sum(ridge[0, :])
    acc_ridge_spam[i] = ridge[1, 1] / sum(ridge[1, :])
    acc_elnet_ham[i] = elnet[0, 0] / sum(elnet[0, :])
    acc_elnet_spam[i] = elnet[1, 1] / sum(elnet[1, :])

# Get mean of accuracy for each of the classes in lasso, ridge, elasticnet methods
print('GLM Lasso Ham','\n',np.mean(acc_lasso_ham))
print('GLM Lasso Spam','\n',np.mean(acc_lasso_spam))
print('GLM Ridge Ham','\n',np.mean(acc_ridge_ham))
print('GLM Ridge Spam','\n',np.mean(acc_ridge_spam))
print('GLM Net Ham','\n',np.mean(acc_elnet_ham))
print('GLM Net Spam','\n',np.mean(acc_elnet_spam))
