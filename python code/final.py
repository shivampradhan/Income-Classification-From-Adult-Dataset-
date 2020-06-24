
# In[1]:


#!pip install joblib
#!pip install pickle


# # import

# In[2]:


from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib
import pickle
import numpy as np
import os
import configparser
import statistics 


# In[3]:


colm=['id', 'age', 'workclass', 'education', 'education_num',
       'marital_status', 'occupation', 'relationship', 'race', 'sex',
       'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']


# ## Load Data by CSV file with header

# In[4]:


path = os.getcwd()


# In[5]:


os.chdir('../')
path


# In[6]:



config = configparser.ConfigParser()
config.read('config.ini')
input_path = config['final_path']['InputPath']
#output_path = config['train_path']['OutputPath']


# In[7]:


data= pd.read_csv(input_path)
#datanh=pd.read_csv(r'C:\Users\Admin\Downloads\data_without_header.csv',header = None)
id=data['id']
id.shape


# # model loading

# In[8]:


os.chdir(os.path.join(os.getcwd(), "model"))


# In[9]:


os.getcwd()


# In[10]:


filename = 'model2.sav'
model = pickle.load(open(filename, 'rb'))

if (filename == 'model2.sav'):
    col= model.get_booster().feature_names
    cols=['id']+col 
    colm=cols[:13]+['native_country']#input data frame
else:
    cols=colm
#reorder the pandas dataframe
#pd_dataframe = pd_dataframe[cols_when_model_builds]


# In[11]:


def proprocess(test):
    
    #test=test
    data=pd.DataFrame(test,columns=cols)
    #print(data)
    data=data.fillna(0)
    
    #data=data.dropna()
    scaler=MinMaxScaler()
    numerical=['age','education_num','capital_gain','capital_loss','hours_per_week']
    data[numerical] = scaler.fit_transform(data[numerical])


    #Changing categorical to ordinal
  #  data['over_50k'] = data['over_50k'].map({'<=50K': 0, '>50K': 1}).astype(int)
    data['sex'] = data['sex'].map({'Male': 0, 'Female': 1}).astype(int)
    data['race'] = data['race'].map({'Black': 0, 'Asian-Pac-Islander': 1,'Other': 2, 'White': 3, 'Amer-Indian-Eskimo': 4}).astype(int)
    data['marital_status'] = data['marital_status'].map({'Married-spouse-absent': 0, 'Widowed': 1, 'Married-civ-spouse': 2, 'Separated': 3, 'Divorced': 4,'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)
    data['workclass'] = data['workclass'].map({'Self-emp-inc': 0, 'State-gov': 1,'Federal-gov': 2, 'Without-pay': 3, 'Local-gov': 4,'Private': 5, 'Self-emp-not-inc': 6}).astype(int)
    data['education'] = data['education'].map({'Some-college': 0, 'Preschool': 1, '5th-6th': 2, 'HS-grad': 3, 'Masters': 4, '12th': 5, '7th-8th': 6, 'Prof-school': 7,'1st-4th': 8, 'Assoc-acdm': 9, 'Doctorate': 10, '11th': 11,'Bachelors': 12, '10th': 13,'Assoc-voc': 14,'9th': 15}).astype(int)
    data['relationship'] = data['relationship'].map({'Not-in-family': 0, 'Wife': 1, 'Other-relative': 2, 'Unmarried': 3,'Husband': 4,'Own-child': 5}).astype(int)
    data['occupation'] = data['occupation'].map(
    { 'Farming-fishing': 1, 'Tech-support': 2, 'Adm-clerical': 3, 'Handlers-cleaners': 4, 
     'Prof-specialty': 5,'Machine-op-inspct': 6, 'Exec-managerial': 7,'Priv-house-serv': 8,'Craft-repair': 9,'Sales': 10, 'Transport-moving': 11, 'Armed-Forces': 12, 'Other-service': 13,'Protective-serv':14}).astype(int)


    data=data.drop('id', axis=1)
    data=pd.get_dummies(data, drop_first=True)
    data=data.fillna(0)
    #print(data)
    return data


# In[12]:


#data=[20733,41,'Private','Some-college',10,'Divorced','Tech-support','Not-in-family','White','Male',0,0,50,'United-States','<=50K']


# In[13]:


X_test=proprocess(data)
X_test


# In[ ]:





# In[14]:


prediction=model.predict(X_test)
prediction
# This is output of one model but we want more generic result


# # Ensemble of ANN,Random Forest and Xgboost

# In[15]:


def Ensembledpredict (data):
    X_test=data
    model1 = pickle.load(open('model1.sav', 'rb'))
    model2 = pickle.load(open('model2.sav', 'rb'))
    model3 = pickle.load(open('model3.sav', 'rb'))
    pred1=model1.predict(X_test)
    pred2=model2.predict(X_test)
    pred3=model3.predict(X_test)
    
    prediction = np.array([])
    for i in range(0,len(X_test)):
        prediction = np.append(prediction, statistics.mode([pred1[i], pred2[i], pred3[i]]))
    return prediction


# # Ensemble is the final model we prefer as it is more generic 

# In[16]:


prediction=Ensembledpredict(X_test)
prediction


# # saving Predicton

# In[17]:


d = {'Id':id,'Prediction':prediction}
df = pd.DataFrame(d)


# In[18]:


os.chdir('../')
os.chdir(os.path.join(os.getcwd(), "data"))


# In[19]:


df.to_csv('prediction.csv',index=False)


# In[20]:


df.head()


# # For manually predicting for a value

# In[21]:


manual_data=[20733,41,'Private','Some-college',10,'Divorced','Tech-support','Not-in-family','White','Male',0,0,50,'United-States']
manual_data=pd.DataFrame([manual_data],columns=colm)
manual_data
mX_test=proprocess(manual_data)
os.chdir('../')
os.chdir(os.path.join(os.getcwd(), "model"))

predicted=Ensembledpredict(mX_test)
predicted


# # for checking accuracy with known target
# def accuracy(a,b,n):
#     l=[]
#     for  i in list(range(n)):
#             if((a[i]-b[i])==0):
#                 l.append(1)
#             else:
#                 l.append(0)
# 
# 
#     accuracy=np.sum(l)/len(l)*100
#     print(accuracy)
#     
#     
# actual= pd.read_csv(r'D:\Shivam\Upwork\JoleneMartin\1\2nd\clean_data.csv')
# actual['over_50k'] = actual['over_50k'].map({'<=50K': 0, '>50K': 1}).astype(int)
# real=actual['over_50k']
# 
# a=list(prediction)
# b=list(real)
# n=len(a)
# accuracy (a,b,n)

# In[ ]:




