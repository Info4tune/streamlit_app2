#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv(r'C:\Users\elise\Desktop\data_hackers\dh_rep.csv')


# In[3]:


df.info()


# In[4]:


#rename the columns that might be used 
df.rename(columns={"('P1', 'age')": "age", "('P2', 'gender')": "gender", "('P8', 'degreee_level')": "degree_level",
                   "('P16', 'salary_range')": "salary", "('P17', 'time_experience_data_science')": "time_experience_ds",
                   "('P18', 'time_experience_before')": "time_experience_before", "('P19', 'is_data_science_professional')": "ds_professional",
                   "('P22', 'most_used_proggraming_languages')": "language", "('P5', 'living_state')": "state",
                   "('P10', 'job_situation')": "job_situation", "('P12', 'workers_number')": "workers_number",
                   "('P13', 'manager')": "manager","('D3', 'anonymized_degree_area')": "degree_area", 
                   "('D4', 'anonymized_market_sector')": "market_sector", "('D6', 'anonymized_role')": "role", "('P21', 'sql_')": "sql","('P21', 'r')": "r",
                   "('P21', 'python')": "python"}, inplace=True)


# In[5]:


df["salary"].value_counts()


# In[6]:


#data manipulation on salary column
variable_split= df["salary"].str.split(" ")
df["salario1"]= variable_split.str.get(2)
df["salario2"]= variable_split.str.get(5)



#split again
variable_split= df["salario1"].str.split("/")
df["salario1"]= variable_split.str.get(0)


variable_split= df["salario2"].str.split("/")
df["salario2"]= variable_split.str.get(0)


# In[7]:


df["salario1"].value_counts()


# In[8]:


#replace the R$ per nan
df.loc[df['salario1'] == "R$",'salario1'] = np.nan


# In[9]:


#remove the "." to transform it into international form
df["salario1"].replace(["1.001"], "1001", inplace= True)
df["salario1"].replace(["2.001"], "2001", inplace= True)
df["salario1"].replace(["3.001"], "3001", inplace= True)
df["salario1"].replace(["4.001"], "4001", inplace= True)
df["salario1"].replace(["6.001"], "6001", inplace= True)
df["salario1"].replace(["8.001"], "8001", inplace= True)
df["salario1"].replace(["12.001"], "12001", inplace= True)
df["salario1"].replace(["16.001"], "16001", inplace= True)

df["salario2"].replace(["2.000"], "2009", inplace= True)
df["salario2"].replace(["3.000"], "3000", inplace= True)
df["salario2"].replace(["4.000"], "4000", inplace= True)
df["salario2"].replace(["6.000"], "6000", inplace= True)
df["salario2"].replace(["8.000"], "8000", inplace= True)
df["salario2"].replace(["12.000"], "12000", inplace= True)
df["salario2"].replace(["20.000"], "20000", inplace= True)


# In[10]:


df["salario1"]= pd.to_numeric(df["salario1"])
df["salario2"]= pd.to_numeric(df["salario2"])       

#take the mean of both salaries
df["salario"]= (df["salario2"]+df["salario1"])/2


# In[11]:


#using just data science professional's data to build the model
ds_prof= df[df.ds_professional ==1]


# In[12]:


#creating a new dataframe with only the relevant variables

ds_prof2 = ds_prof.loc[:, ["age", "degree_level", "time_experience_ds", "job_situation",
               "degree_area", "salario", "state", "workers_number", "manager", "market_sector", "role", "sql",
                          "r", "python"]] 

#variables not chosen for predictive power but for easiness of data selection/willingness to share in a app


# In[13]:


ds_prof2.info()


# In[14]:


ds_prof2["degree_level"].value_counts()


# In[15]:


ds_prof2["time_experience_ds"].value_counts()


# In[16]:


ds_prof2["job_situation"].value_counts() 


# In[17]:


ds_prof2['degree_area'].value_counts() 


# In[18]:


ds_prof2['state'].value_counts() 


# In[19]:


ds_prof2['manager'].value_counts()


# In[20]:


ds_prof2['market_sector'].value_counts() 


# In[21]:


ds_prof2['time_experience_ds'].value_counts() 


# In[22]:


ds_prof2['age_bins'] = pd.cut(x= ds_prof2['age'], bins = [18, 24, 30, 40, 50]) #creating bins of variable age
ds_prof2


# In[23]:


ds_prof2['workers_number'].value_counts() 


# In[24]:


filters = [
    (ds_prof2.workers_number == "de 1 a 5"),
    (ds_prof2.workers_number == "de 6 a 10"),
    (ds_prof2.workers_number == "de 11 a 50"), 
    (ds_prof2.workers_number == "de 51 a 100"),
    (ds_prof2.workers_number == "de 101 a 500"),
    (ds_prof2.workers_number == "de 501 a 1000"), 
    (ds_prof2.workers_number == "de 1001 a 3000"),
    (ds_prof2.workers_number == "Acima de 3000")
]

values = ["Micro", "Micro", "Pequeno", "Médio", "Grande", "Grande", "Grande", "Grande"] #creating a new variable classifying company´s sizes


# In[25]:


ds_prof2['porte'] = np.select(filters, values, np.nan)
ds_prof2.head()


# In[26]:


ds_prof2['porte'].replace('nan', np.nan, inplace= True) #forcing the nans to be actual nans since the previous functions doesn´t do the trick


# In[27]:


ds_full = ds_prof2.copy()
ds_full.drop(['age', 'workers_number'], axis=1, inplace=True)


# In[28]:


ds_full['state'] = ds_full['state'].fillna('Outro')



# In[35]:



from pycaret.regression import *

s= setup(ds_full, target = 'salario',
        combine_rare_levels = True,
        ordinal_features = {'porte' : ["Micro", "Pequeno", "Médio", "Grande"],
                           'time_experience_ds' : ['Não tenho experiência na área de dados','Menos de 1 ano', 
                                                   'de 1 a 2 anos', 'de 2 a 3 anos',
                                                   'de 4 a 5 anos', 'de 6 a 10 anos', 'Mais de 10 anos']})


# In[36]:


best = compare_models()
print(best) #models are no good but since was previously decided to use some predefined var  for the lead app, this will do 


# In[37]:



save_model(best, 'salario_ds_best')


ds_full.head()




# %%
