
# # data loading

import numpy as np
import pandas as pd
import seaborn as sns

import configparser

pd.options.mode.chained_assignment = None 
np.random.seed(30)

config = configparser.ConfigParser()
config.read('config.ini')
input_path = config['path']['InputPath']
output_path = config['path']['OutputPath']

raw_data = pd.read_csv(input_path)
print(raw_data.shape)
raw_data.head()


raw_data.isna().sum(axis = 0)


# ## Checking and removing duplicate rows as it wont help the model train

duplicate=raw_data[raw_data.duplicated(subset=None, keep='first')]
print(duplicate.shape)
#duplicate

unique_data=raw_data.drop_duplicates( keep='first', inplace=False)
print(unique_data.shape)
#unique_data


# ## Checking na


unique_data.isnull().sum(axis = 0)


# ## Removing data with na in over_50k as we cant fill it because it is the Target variable


output_na=unique_data[unique_data['over_50k'].isna()]
output_na['over_50k'].isna().count()


input_data=unique_data[~unique_data['over_50k'].isna()]
input_data.shape


# ## so actual  input data we have is 26277 rows

# # correcting wrong label for white in race



print(input_data.groupby('race')['id'].count())



#renaming whi t e 
#input_data.loc[input_data.race == 'Whi t e','Whi t e'] = 'White'
input_data['race'][input_data.race == 'Whi t e'] = 'White'

print(input_data.groupby('race')['id'].count())


# ## dealing with wrong data of age -1
# 

sns.distplot(input_data.age)


c=input_data[input_data['age']<0]['age'].count()
print(f"Age less than zero in dataset {c}")



input_data[(input_data['age']<22) & (input_data['over_50k']=='<=50K')]['age'].count()



x=input_data[(input_data['age']<22) & (input_data['over_50k']=='<=50K')]
#x.groupby('workclass').count() 2172
#x.groupby('marital_status').count() 2672
x.groupby('marital_status').count()
#y=input_data[(input_data['marital_status']=='Never-married')]
#list(y.groupby('age')['id'].count())


# ### most of the na in age are never  married 

c=input_data[input_data['age']<0]['age'].count()
print(f"Age less than zero in dataset {c}")


# # age imputation of never married uniformly



x=((input_data['age'] == -1) & (input_data['marital_status']=='Never-married'))

input_data.loc[((input_data['age'] == -1) & (input_data['marital_status']=='Never-married')),'age'] =np.random.randint(15,35, len(input_data.loc[((input_data['age'] == -1) & (input_data['marital_status']=='Never-married'))]))
sns.distplot(input_data.age)



c=input_data[input_data['age']<0]['age'].count()
print(f"Age less than zero in dataset {c}")



input_data['age'][input_data.age == -1] = np.nan
input_data['age'].isna().sum()


input_data=input_data[~input_data['age'].isna()]
input_data.shape


# ## Dealing with ? in data


input_data['native_country'][input_data.native_country == '?'] = np.nan
input_data['workclass'][input_data.workclass == '?'] = np.nan
input_data['occupation'][input_data.occupation == '?'] = np.nan




workclass_q=input_data[input_data['workclass']=='?']['id'].count()
native_country_q=input_data[input_data['native_country']=='?']['id'].count()
occupation_q=input_data[input_data['occupation']=='?']['id'].count()
print (f"? in workclass is {workclass_q} native_country is {native_country_q} occupation is {occupation_q}")



# # total actual null 



input_data.isna().sum().sum()




input_data[
    (input_data['occupation'].isnull())
           |  (input_data['workclass'].isnull() ) 
           |  (input_data['native_country'].isnull())
         
           |  (input_data['education'].isnull())
        
]['id'].count() # it means there are rows with more than one variable having nan





input_data[
    ((
        input_data['occupation'].isnull())
           |  (input_data['workclass'].isnull() ) 
           |  (input_data['native_country'].isnull())
        
           |  (input_data['education'].isnull())
          
    )
    
           & (input_data['over_50k']=='<=50K')
]['id'].count()


# # dropping the null data ROWS  of over_50k not true area to help solving data imbalance



print(input_data['id'].count())
dropped_data=input_data[~
    ((
        (input_data['occupation'].isnull())
           |  (input_data['workclass'].isnull() )
           |  (input_data['native_country'].isnull())
           |  (input_data['education'].isnull())
    )
           
        
        & (input_data['over_50k']=='<=50K'))
]
dropped_data['id'].count()


# # dealing with remaining NA



dropped_data[
    (dropped_data['education'].isnull()) 
           |  (dropped_data['workclass'].isnull() ) 
           |  (dropped_data['native_country'].isnull())
          
           |  (dropped_data['occupation'].isnull())
           #| (dropped_data['age']<0)
]['id'].count()



workclass_q=dropped_data[dropped_data['workclass'].isnull()]['id'].count()
native_country_q=dropped_data[dropped_data['native_country'].isnull()]['id'].count()
occupation_q=dropped_data[dropped_data['occupation'].isnull()]['id'].count()
print (f"NA in workclass is {workclass_q} native_country is {native_country_q} occupation is {occupation_q}")


# ## Dropping off the common row in workclass and occupation NA

dropped_data[
            (   dropped_data['occupation'].isnull())
           & (dropped_data['workclass'].isnull() ) 
          ]['id'].count()



print(input_data['id'].count())
dropped_data=dropped_data[~
    ((   dropped_data['occupation'].isnull())
           & (dropped_data['workclass'].isnull() ) )
]
dropped_data['id'].count()


workclass_q=dropped_data[dropped_data['workclass'].isnull()]['id'].count()
native_country_q=dropped_data[dropped_data['native_country'].isnull()]['id'].count()
occupation_q=dropped_data[dropped_data['occupation'].isnull()]['id'].count()
print (f"NA in workclass is {workclass_q} native_country is {native_country_q} occupation is {occupation_q}")




# # now we will fill the remaining nan


print("nan left",dropped_data.isna().sum().sum())
mod=dropped_data['native_country'].mode()
dropped_data['native_country'][dropped_data.native_country.isnull()] = 'United-States'
print("nan left",dropped_data.isna().sum().sum())

dropped_data = dropped_data.fillna(dropped_data['education'].value_counts().index[0])
print("nan left",dropped_data.isna().sum().sum())


data=dropped_data
data.shape


# # saving data



data.to_csv(output_path,index=False)




