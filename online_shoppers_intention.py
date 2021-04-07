#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df= pd.read_csv('online_shoppers_intention.csv')
print(df.shape)


# In[3]:


df.isna().sum()


# In[4]:


df.head()


# In[5]:


sns.heatmap(df.corr())


# In[6]:


dft= df.iloc[:,:15]


# In[7]:


mon= dft['Month']
dft.drop({'Month'}, axis=1, inplace=True)


# In[8]:


col= dft.columns


# In[9]:


from sklearn.preprocessing import PowerTransformer
pt= PowerTransformer()


# In[10]:


dft = pt.fit_transform(dft)


# In[11]:


dft= pd.DataFrame(dft, columns=col)


# In[12]:


dft.head()


# In[13]:


dfr= df[['Month', 'VisitorType', 'Weekend', 'Revenue']]
dfr.head()


# In[14]:


from sklearn.preprocessing import LabelEncoder


# In[15]:


for i in dfr.columns:
    dfr[i] = LabelEncoder().fit_transform(dfr[i])


# In[16]:


dfr.head()


# In[17]:


dft= dft.join(dfr)


# In[18]:


dft.head()


# In[19]:


# Lets try K means clustering first.
import scipy.cluster.hierarchy as sch
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[20]:


rev= dft['Revenue']
dft.drop({'Revenue'}, axis=1, inplace=True)


# In[21]:


we= []
sc= []
for i in range(2, 15):
    km= KMeans(n_clusters=i, random_state=42)
    km.fit(dft)
    we.append(km.inertia_)
    lab= km.labels_
    sc.append(silhouette_score(dft,lab))


# In[22]:


plt.plot(range(2,15), we, 'r')
plt.xlabel('clusters')
plt.ylabel('SSD')
plt.show()


# In[23]:


# For KMeans optimum number of cluster is 3


# In[24]:


plt.plot(range(2,15), sc, 'b')
plt.xlabel('Sil score')
plt.ylabel('SSD')
plt.show()


# In[25]:


dft.shape


# In[26]:


km= KMeans(n_clusters=2, random_state=42)
km.fit(dft.iloc[:,:17])


# In[27]:


lab= km.labels_
dft['km_cl'] = lab


# In[28]:


# Agglomerative 

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(dft, method='ward'))


# In[29]:


# For Agglo, optimum number of cluster is 2


# In[30]:


model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
model.fit(dft)
labels = model.labels_


# In[31]:


dft['agg_cl'] = labels
dft['real'] = rev

pd.crosstab(dft['agg_cl'], dft['real'])


# In[32]:


# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[33]:


pca = PCA()
pca.fit(dft.iloc[:,:17])


# In[34]:


pca.explained_variance_


# In[35]:


np.cumsum(pca.explained_variance_ratio_)


# In[36]:


# 10 features are explaining over 93%

pca = PCA(n_components=10)
pca.fit(dft.iloc[:,:17])
X_pca = pca.transform(dft.iloc[:,:17])
X_pca.shape


# In[37]:


X_pca


# In[38]:


xpca= pd.DataFrame(X_pca, columns=[i for i in range(10)])


# In[39]:


xpca


# In[40]:


# Lets apply agglo clustering on this now. 
dendrogram = sch.dendrogram(sch.linkage(xpca, method='ward'))


# In[41]:


# For Agglo, optimum number of cluster is 3

# but lets try with 2 first


# In[42]:


model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
model.fit(xpca)
labels = model.labels_


# In[43]:


xpca['agg_cl'] = labels
xpca['real'] = rev

pd.crosstab(xpca['agg_cl'], xpca['real'])


# In[45]:


xpca


# In[46]:


sns.pairplot(dft)

