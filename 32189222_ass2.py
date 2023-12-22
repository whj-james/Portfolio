#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-success">
#     
# # FIT5196 Assessment 2
# #### Student Name: Huangjin Wang
# #### Student ID: 32189222
# 
# Date: 30-09-2022
# 
# 
# Environment: Python 3.9
# 
# Libraries used:
# * datetime (get date)
# * pandas (for data manipulation) 
# * numpy (for calculating IQR)
# * matplotlib.pyplot (for ploting)
# * networkx (for calculating shortest distance)
# * sklearn.linear_model (for building linear regression model)
# </div>

# <div class="alert alert-block alert-danger">
#     
# ## Table of Contents
# 
# </div>    
# 
# [1. Solution for dirty data](#dirty) <br>
# [2. Imputing missing file](#missing) <br>
# [3. Outlier processing](#outlier) <br>
# [4. Output](#output) <br>

# <div class="alert alert-block alert-warning">
# 
# ## 1.  Solution for dirty data  <a class="anchor" name="dirty"></a>
#     
# </div>

# ### Import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas import Series,DataFrame
import networkx as nx
from sklearn.linear_model import LinearRegression


# In[2]:


dirty = pd.read_csv('32189222_dirty_data.csv')
# have a glance of the data
dirty.info()


# In[3]:


dirty.head()


# ###  Clean date
# Firstly, check the wrong date in the data.

# In[4]:


for date in dirty.date: 
    try: # check if the dates are in the format of YYYY-MM-DD
        datetime.strptime(date, '%Y-%m-%d')
    except: # print out the date, if date format is incorrect
        print(date)
# Reference
#https://theprogrammingexpert.com/check-if-string-is-date-in-python/#:~:text=You%20can%20check%20if%20a,the%20datetime%20object%20is%20created.


# We can see the wrong date in the data. Then we can replace them by correct format.

# In[5]:


dirty.date.replace({'2021-27-02': '2021-02-27',
                   '2021-21-07': '2021-07-21',
                   '2021-17-09': '2021-09-17',
                   '2021-21-12': '2021-12-21',
                   '2021-Feb-04': '2021-02-04',
                   '2021-20-01': '2021-01-20'},inplace = True)


# In[6]:


# double check the date column
for date in dirty.date: 
    try:# check if the dates are in the format of YYYY-MM-DD
        datetime.strptime(date, '%Y-%m-%d')
    except:# print out the date, if date format is incorrect
        print(date)


# Now these wrong dates are imputed to correct format.

# ### Clean order type

# Change all the order type based on the order time, this will replace possible wrong order type.

# In[7]:


for index,row in dirty.iterrows():
    if row.time <= '12:00:00': 
        # if time is earlier than 12:00:01, it is breakfast
        dirty.loc[index,'order_type'] = 'Breakfast'
        continue
    if row.time >= '12:00:01' and row.time <='16:00:00' :
        # if time is later than 12:00:00 and earlier than 16:00:01, it is lunch
        dirty.loc[index,'order_type'] = 'Lunch'
        continue
    if row.time >= '16:00:01' and row.time <='20:00:00':
        # if time is later than 16:00:00 and earlier than 20:00:01, it is dinner
        dirty.loc[index,'order_type'] = 'Dinner'
        continue


# ### Clean branch_code

# In[8]:


# check branch_code values
dirty.branch_code.value_counts()


# From the values of branch code, we can see that values are inconsistent. e.g. `'TP'` and `'tp'`, `'NS'` and `'ns'`, `'BK'` and `'bk'`.

# In[9]:


# Upper all the values in branch code
dirty.branch_code = dirty['branch_code'].apply(lambda code: code.upper())


# In[10]:


# check values after cleaning
dirty.branch_code.value_counts()


# In[11]:


pd.crosstab(dirty.order_id.str.slice(0,4),dirty.branch_code)


# However, from the cross table of order_id and branch_code, the first four letters of order id is related to the branch code. e.g. `ORDA` relat to `BK`. Thus, There are some wrong relations in our data, `ORDA` has 2 incorrect values of `TP`, `ORDB` has an incorrect value `BK`, `ORDK` has an incorrect value `TP`, `ORDC` has an incorrect value `BK`.
# 
# We just need to change all branch code values based on their first four letters of order id.

# In[12]:


for index,row in dirty.iterrows():
    if dirty.order_id[index].startswith('ORDA'):
        # if it starts with ORDA, change branch code to BK
        dirty.loc[index,'branch_code'] = 'BK'
        continue
    if dirty.order_id[index].startswith('ORDB'):
        # if it starts with ORDB, change branch code to TP
        dirty.loc[index,'branch_code'] = 'TP'
        continue
    if dirty.order_id[index].startswith('ORDC'):
        # if it starts with ORDC, change branch code to NS
        dirty.loc[index,'branch_code'] = 'NS'
        continue
    if dirty.order_id[index].startswith('ORDK'):
        # if it starts with ORDK, change branch code to TP
        dirty.loc[index,'branch_code'] = 'TP'
        continue


# In[13]:


# check cross table after cleaning
pd.crosstab(dirty.order_id.str.slice(0,4),dirty.branch_code)


# Now, all the order_ids have their correct branch codes.

# ### Clean customer lat and lon

# In[14]:


dirty.customer_lat.describe()


# In[15]:


dirty.customer_lon.describe()


# By using `describe` funciton, we can see some values in customer lat are actually customer lon. However, some values in customer lon are customer lat. e.g. most of customer lat are negative numbers, but there are some positive numbers in customer lat. And most of customer lon are positive numbers also invovles negative numbers.

# In[16]:


wrong_lat = []
wrong_lon = []
for index, row in dirty.iterrows():
    if dirty.loc[index,'customer_lat'] > 0:
        wrong_lat.append(index)
    if dirty.loc[index,'customer_lon'] < 0 :
        wrong_lon.append(index)


# In[17]:


wrong_lat


# In[18]:


wrong_lon


# In[19]:


dirty.iloc[wrong_lat,[7,8]]


# From these wrong records, we can see that some values are swaped between `lat` and `lon`, like row 16. However, some values in lat should be negative such as `37.805444`, as others are negative 37 something. Thus, what we need to do is swap the lat and lon for these rows and change values to negative for those who are positive. For swaping lat and lon, we need to make two another copies of origional data to change value. 

# In[20]:


sub_lat = dirty.iloc[wrong_lon,7].copy() # copy of origional lat data
sub_lon = dirty.iloc[wrong_lon,8].copy() # copy of origional lon data


# In[21]:


sub_lat


# In[22]:


sub_lon


# In[23]:


dirty.iloc[wrong_lon,7] = sub_lon # swap lat and lon 
dirty.iloc[wrong_lon,8] = sub_lat # swap lon and lat


# In[24]:


dirty.iloc[wrong_lat,[7,8]] # check lon after swap


# Now, there is one more step to go, change positive lat to negative. 

# In[25]:


# change all the lat value to positive and mutiply -1
dirty.customer_lat = abs(dirty.customer_lat) *-1


# In[26]:


dirty.iloc[wrong_lat,[7,8]] # check after cleaning


# Now, all customer lat and lon values are in the correct format.

# ### Clean distance to customer

# Since we have lat and lon for both customer and branch, we can use Dijkstra algorithm to calculate the shorest distance between customer and branch. Thus, we can validate if values are correct. First, we need to know node for branches and customers, then create a graph to store every node and edge, and implement dijkstra algorithm to calculate the shortest distance between order branches and customers eventually.

# In[27]:


nodes = pd.read_csv('nodes.csv') # read nodes 
edges = pd.read_csv('edges.csv') # read edges
branches = pd.read_csv('branches.csv') # read branches


# In[28]:


nodes.head()


# In[29]:


edges.head()


# In[30]:


branches.head()


# We already have branches and corresponding lat and lon. Thus,we can find their node.

# In[31]:


for index,row in nodes.iterrows():
    if row.lat == branches.loc[0,'branch_lat'] and row.lon == branches.loc[0,'branch_lon']:
        NS_node = int(row.node)
        continue
    if row.lat == branches.loc[1,'branch_lat'] and row.lon == branches.loc[1,'branch_lon']:
        TP_node = int(row.node)
        continue
    if row.lat == branches.loc[2,'branch_lat'] and row.lon == branches.loc[2,'branch_lon']:
        BK_node = int(row.node)
        continue


# In[32]:


# join branch code and their lat,lon and node together
branch_node = pd.concat([branches,DataFrame({'node': [NS_node,TP_node,BK_node]})],axis = 1)
branch_node


# In[33]:


# remove unnecessary column and remove columns
branch_node = branch_node.drop('branch_name',axis = 1)
branch_node.columns =['branch_code','branch_lat', 'branch_lon', 'branch_node']


# In[34]:


branch_node


# Now, we have branch code and corresponding lat, lon and node. Next, we will get node for each customer.

# In[35]:


# get customers' lat and lon and order branch
customer = dirty[['order_id','customer_lat','customer_lon','branch_code']]
customer.columns = ['order_id', 'lat', 'lon','branch_code']
customer


# In[36]:


# find customer's node by joining nodes data frame and customer data frame
customer_node = customer.merge(nodes,how = 'left', on = ['lat','lon'])
# rename columns for later process
customer_node.columns = ['order_id', 'cus_lat','cus_lon','branch_code', 'cus_node']
customer_node


# We have nodes for both customers and branches in different data frame. Then join two data frame for implementing shortest distance calculation.

# In[37]:


# Join two dataframes by common column 'branch_code'
all_node = customer_node.merge(branch_node,how = 'left', on = 'branch_code')
all_node


# Then we need to create a graph to calculate the shortest distance.

# In[38]:


# Crate a graph to for calculating shortest path
graph = nx.Graph()
# Add nodes
for node in nodes.node:
    graph.add_node(node)

# Add edges
for index,row in edges.iterrows():
    graph.add_edge(row.u,row.v, weight = row['distance(m)']) #(first node, second node, weight)

    
# Reference
# https://networkx.org/documentation/stable/reference/classes/graph.html


# In[39]:


short_distance = [] # calculated shortest distance used for updating wrong distance
wrong_distance = [] # index of wrong distance in the data
# Calculate shortest distance by dijstra algorithm
for index,row in all_node.iterrows():
    distance = nx.algorithms.shortest_paths.generic.shortest_path_length(graph,
                                                                        source = row['cus_node'],
                                                                        target = row['branch_node'],
                                                                         weight = 'weight',
                                                                        method = 'dijkstra')
    distance = round(distance/1000,3) # change m to km
    short_distance.append(distance)
    # if recorded distance is wrong, then record its index
    if distance != dirty.loc[index, 'distance_to_customer_KM']: 
        wrong_distance.append(index)
# Reference
#https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path_length.html


# In[40]:


# Show the calculated distance
short_distance_df = pd.DataFrame(short_distance, columns = ['short_distance'])
short_distance_df


# In[41]:


# show the row indexs which values is wrong
wrong_distance


# In[42]:


# Show the wrong distance values
dirty.iloc[wrong_distance,-2]


# In[43]:


# change the wrong distance by calculated shortest distance
dirty.iloc[wrong_distance,-2] = short_distance_df.iloc[wrong_distance,0]


# In[44]:


# show the wrong part after cleaning
dirty.iloc[wrong_distance,-2]


# Now, all the values of shortest distance to customer are correct.

# # Clean customer loyalty

# We can use delivery fee to validate if customer has loyalty since delivery fee column is error free. However, delivery fee is influenced by some other factors such as weekend or weekday and time of the day. We need to get these two factors for each order first. Then we can use delivery fee to calculate the 

# In[45]:


weekend = [] # to store if the date is weekend or not
for index, row in dirty.iterrows():
    # get date
    date = datetime.strptime(row.date, '%Y-%m-%d')
    if date.isoweekday() <= 5: # if day is 1,2,3,4,5, it is the weekday
        weekend.append('0')
    else: # otherwise, it is the weekend
        weekend.append('1')


# In[46]:


weekend = pd.DataFrame(weekend, columns = ['is_weekend']) # concat dataframes for later process
weekend


# In[47]:


dirty = pd.concat([dirty, weekend], axis =1) # concat dataframes for later process
dirty


# In[48]:


time_of_the_day = [] # to store time of the day
for index, row in dirty.iterrows():
    if row.order_type == 'Breakfast': # if it is Morning, then 0
        time_of_the_day.append('0')
        continue
    if row.order_type == 'Lunch':   # if it is Afternoon, then 1
        time_of_the_day.append('1')
        continue
    if row.order_type == 'Dinner': # if it is Evevning, then 2
        time_of_the_day.append('2')
        continue


# In[49]:


time_of_the_day = pd.DataFrame(time_of_the_day, columns = ['time_of_the_day'])
time_of_the_day


# In[50]:


dirty = pd.concat([dirty, time_of_the_day], axis =1) # concat dataframes for later process
dirty


# Now, we need to use missing_date to create a linear regression model as it contains correct data. Then, the missing data will be the training set, and apply the prediction to the delivery fee of dirty data.

# In[51]:


missing = pd.read_csv('32189222_missing_data.csv')
missing.info()


# As missing data have NA value, we need to drop the NA value.

# In[52]:


missing.info()


# In[53]:


# drop Na value for delivery fee column
missing = missing.dropna(subset = ['delivery_fee'], axis = 0) 
missing = missing.reset_index(drop = True)
missing


# Also, we need to add those information,if it is weekend and time of the day to the dataframe.

# In[54]:


weekend = [] # to store if the date is weekend or not
for index, row in missing.iterrows():
    # get date
    date = datetime.strptime(row.date,'%Y-%m-%d')
    if date.isoweekday() <= 5: # if day is 1,2,3,4,5, it is the weekday
        weekend.append('0')
    else: # otherwise, it is the weekend
        weekend.append('1')
time_of_the_day = [] # to store time of the day
for index, row in missing.iterrows():
    if row.order_type == 'Breakfast': # if it is Morning, then 0
        time_of_the_day.append('0')
        continue
    if row.order_type == 'Lunch':   # if it is Afternoon, then 1
        time_of_the_day.append('1')
        continue
    if row.order_type == 'Dinner': # if it is Evevning, then 2
        time_of_the_day.append('2')
        continue


# In[55]:


time_of_the_day = pd.DataFrame(time_of_the_day, columns = ['time_of_the_day'])
weekend = pd.DataFrame(weekend, columns = ['is_weekend'])

missing = pd.concat([missing, weekend,time_of_the_day],axis =1) # concat dataframe
missing.head()


# Now, we can build linear regression model for each branch, first we need to split missing data by branch code before we can do build the LR model.

# In[56]:


bk_missing = missing[missing.branch_code == 'BK'] # For which branch code is bk
tp_missing = missing[missing.branch_code == 'TP'] # For which branch code is tp
ns_missing = missing[missing.branch_code == 'NS'] # For which branch code is ns


# Now, we can build LR model for delivery fee based on difference branch.

# In[57]:


bk_lm = LinearRegression() 
tp_lm = LinearRegression()
ns_lm = LinearRegression()


# In[58]:


# for each branch
# make prediction for delivery fee based on is weekend, time of the day, and distance to customer
bk_lm.fit(bk_missing[['is_weekend','time_of_the_day','distance_to_customer_KM']], 
          bk_missing['delivery_fee'])
tp_lm.fit(tp_missing[['is_weekend','time_of_the_day','distance_to_customer_KM']], 
          tp_missing['delivery_fee'])
ns_lm.fit(ns_missing[['is_weekend','time_of_the_day','distance_to_customer_KM']], 
          ns_missing['delivery_fee'])


# In[59]:


# create an empty column to store predicted fee later
dirty['predict_fee'] = 0


# In[60]:


# split dirty data based on different branch
# Since we need to build LR model for each branch
bk_dirty = dirty[dirty.branch_code == 'BK']
tp_dirty = dirty[dirty.branch_code == 'TP']
ns_dirty = dirty[dirty.branch_code == 'NS']


# In[61]:


# for each branch
# Make prediction for dirty data based on training data set, which is the missing data
bk_dirty['predict_fee'] = bk_lm.predict(bk_dirty[['is_weekend','time_of_the_day','distance_to_customer_KM']])
tp_dirty['predict_fee'] = tp_lm.predict(tp_dirty[['is_weekend','time_of_the_day','distance_to_customer_KM']])
ns_dirty['predict_fee'] = ns_lm.predict(ns_dirty[['is_weekend','time_of_the_day','distance_to_customer_KM']])


# Now, we have the predicted fee for each branch, we can add these predicted value to the dirty data.

# In[62]:


# based on the index, add predicted fee for each record
for index,row in bk_dirty.iterrows():
    dirty.iloc[index,-1] = row.predict_fee
for index,row in tp_dirty.iterrows():
    dirty.iloc[index,-1] = row.predict_fee
for index,row in ns_dirty.iterrows():
    dirty.iloc[index,-1] = row.predict_fee


# In[63]:


dirty.head() # check after prediction


# In[64]:


wrong_loyalty = []
for index,row in dirty.iterrows():
    if row.predict_fee * 0.5 < row.delivery_fee  and row['customerHasloyalty?'] == 1:
        wrong_loyalty.append(index)
        continue
    if row.predict_fee *0.5 > row.delivery_fee and row['customerHasloyalty?'] == 0:
        wrong_loyalty.append(index)


# In[65]:


dirty.iloc[wrong_loyalty,[-6,-4,-1]]


# Since we predicted the delivery fee, it may have som bias between real fee, we need to filter this wrong index again, which indexs we just got. Also, we can see that from the wrong customer loyalty, all value is 1, so we need to remove indexs that half of the predicted fee has small difference when compare to real delivery fee.

# In[66]:


new_wrong_loyalty = []
for i in wrong_loyalty:
    predict = dirty.loc[i,'predict_fee']
    true = dirty.loc[i,'delivery_fee']
    if abs(predict*0.5 - true) > 2:
        new_wrong_loyalty.append(i)


# In[67]:


dirty.iloc[new_wrong_loyalty,[-6,-4,-1]]


# Now, we have our final wrong record, since the difference between predict_fee and delivery_fee is small, which can be caused by prediction bias. And apparently these customers are not loyal.

# In[68]:


dirty.iloc[new_wrong_loyalty,-6] = 0


# In[69]:


dirty.iloc[new_wrong_loyalty,-6] # check after cleaning


# Thus, the cleaning for dirty_data file is done.

# <div class="alert alert-block alert-warning">
#     
# ## 2.  Imputing missing file  <a class="anchor" name="missing"></a>
#  </div>

# In[70]:


missing = pd.read_csv('32189222_missing_data.csv') # read missing data again


# In[71]:


missing.head()


# In[72]:


missing.info()


# As we can see, there should be 500 rows in the missing file, however, the branch_code has only 450 rows, and delivery_fee has only 400 rows. For branch_code, we can fill by its order_id for each record, and we will use the same linear regression model used in dirty_data to predict delivery fee.

# In[73]:


pd.crosstab(missing.order_id.str.slice(0,4),missing.branch_code)


# By using cross table, we can see the first four letters of order id, and their corresponding branch code.

# In[74]:


for index,row in missing.iterrows():
    if missing.order_id[index].startswith('ORDA'):
        # if it starts with ORDA, change branch code to BK
        missing.loc[index,'branch_code'] = 'BK'
        continue
    if missing.order_id[index].startswith('ORDB'):
        # if it starts with ORDB, change branch code to TP
        missing.loc[index,'branch_code'] = 'TP'
        continue
    if missing.order_id[index].startswith('ORDC'):
        # if it starts with ORDC, change branch code to NS
        missing.loc[index,'branch_code'] = 'NS'
        continue
    if missing.order_id[index].startswith('ORDK'):
        # if it starts with ORDK, change branch code to TP
        missing.loc[index,'branch_code'] = 'TP'
        continue
    if missing.order_id[index].startswith('ORDI'):
        # if it starts with ORDI, change branch code to NS
        missing.loc[index,'branch_code'] = 'NS'
        continue
    if missing.order_id[index].startswith('ORDJ'):
        # if it starts with ORDJ, change branch code to TP
        missing.loc[index,'branch_code'] = 'TP'
        continue
    if missing.order_id[index].startswith('ORDX'):
        # if it starts with ORDX, change branch code to BK
        missing.loc[index,'branch_code'] = 'BK'
        continue
    if missing.order_id[index].startswith('ORDY'):
        # if it starts with ORDY, change branch code to TP
        missing.loc[index,'branch_code'] = 'TP'
        continue
    if missing.order_id[index].startswith('ORDZ'):
        # if it starts with ORDZ, change branch code to NS
        missing.loc[index,'branch_code'] = 'NS'
        continue


# In[75]:


missing.info()


# Now, all the missing branch code has filled. For predict delivery fee, we also need other information sush as, is weekend or not and time of the day. We wil do the same process as above to add two columns in the missing data.

# In[76]:


weekend = [] # to store if the date is weekend or not
for index, row in missing.iterrows():
    # get date
    date = datetime.strptime(row.date,'%Y-%m-%d')
    if date.isoweekday() <= 5: # if day is 1,2,3,4,5, it is the weekday
        weekend.append('0')
    else: # otherwise, it is the weekend
        weekend.append('1')
time_of_the_day = [] # to store time of the day
for index, row in missing.iterrows():
    if row.order_type == 'Breakfast': # if it is Morning, then 0
        time_of_the_day.append('0')
        continue
    if row.order_type == 'Lunch':   # if it is Afternoon, then 1
        time_of_the_day.append('1')
        continue
    if row.order_type == 'Dinner': # if it is Evevning, then 2
        time_of_the_day.append('2')
        continue


# In[77]:


time_of_the_day = pd.DataFrame(time_of_the_day, columns = ['time_of_the_day'])
weekend = pd.DataFrame(weekend, columns = ['is_weekend'])

missing = pd.concat([missing, weekend,time_of_the_day],axis =1) # concat dataframe
missing.head()


# Before building LR model, we need to split data for each branch and predict for each branch.

# In[78]:


# get rows that have missing value.
fee_missing = missing[missing.delivery_fee.isnull()==True]
fee_missing.head()


# In[79]:


bk_missing = fee_missing[fee_missing.branch_code == 'BK'] # For branch code is bk
tp_missing = fee_missing[fee_missing.branch_code == 'TP'] # For branch code is tp
ns_missing = fee_missing[fee_missing.branch_code == 'NS'] # For branch code is ns


# In[80]:


# for each branch
# Make prediction for dirty data based on training data set, which is the missing data
bk_missing['delivery_fee'] = bk_lm.predict(bk_missing[['is_weekend','time_of_the_day','distance_to_customer_KM']])
tp_missing['delivery_fee'] = tp_lm.predict(tp_missing[['is_weekend','time_of_the_day','distance_to_customer_KM']])
ns_missing['delivery_fee'] = ns_lm.predict(ns_missing[['is_weekend','time_of_the_day','distance_to_customer_KM']])


# In[81]:


# based on index fill delivery fee by prediction for each branch
for index,row in bk_missing.iterrows():
    missing.loc[index,'delivery_fee'] = row.delivery_fee
for index,row in tp_missing.iterrows():
    missing.loc[index,'delivery_fee'] = row.delivery_fee
for index,row in ns_missing.iterrows():
    missing.loc[index,'delivery_fee'] = row.delivery_fee


# In[82]:


missing.info()


# Now, imputation for missing_data file is done.

# <div class="alert alert-block alert-warning">
#     
# ## 3.  Outlier processing  <a class="anchor" name="outlier"></a>
#  </div>

# Removing outliers by using IQR technique. In the descriptive statistics, any numbers are smaller than `quartile 1` - 1.5 *IQR and larger than `quartile 3` +1.5 *IQR will be considered as outliers.

# In[83]:


# read file
outlier = pd.read_csv('32189222_outlier_data.csv')


# In[84]:


outlier.delivery_fee.describe()


# In[85]:


plt.boxplot(outlier.delivery_fee) # boxplot for delivery fee


# We can see that some data are out of the upper fence and lower fence, these data are outliers.

# In[86]:


# https://www.statology.org/interquartile-range-python/#:~:text=The%20interquartile%20range%2C%20often%20denoted,75th%20percentile)%20of%20a%20dataset.
q3 = np.percentile(outlier.delivery_fee, 75)
q1 = np.percentile(outlier.delivery_fee, 25)
IQR = q3 -q1
print('IQR is ' +str(IQR))


# In[87]:


upper_fence = q3 + 1.5*IQR
lower_fence = q1 - 1.5*IQR
print('upper_fence is ' + str(upper_fence))
print('lower_fence is ' + str(lower_fence))


# In[88]:


# remove outliers
for index, row in outlier.iterrows():
    if row.delivery_fee > upper_fence or row.delivery_fee < lower_fence:
        outlier = outlier.drop(index = index)


# In[89]:


outlier.info() # Check after removing outliers


# <div class="alert alert-block alert-warning">
#     
# ## 4.  Output <a class="anchor" name="output"></a>
#  </div>

# In[90]:


# Remove unnecessary columns
dirty = dirty.drop(['is_weekend','time_of_the_day','predict_fee'],axis = 1)
missing = missing.drop(['is_weekend','time_of_the_day'],axis = 1)


# In[91]:


dirty.to_csv('32189222_dirty_data_solution.csv',index = False)
missing.to_csv('32189222_missing_data_solution.csv',index = False)
outlier.to_csv('32189222_outlier_data_solution.csv', index = False)


# In[ ]:




