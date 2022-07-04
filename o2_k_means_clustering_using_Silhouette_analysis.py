#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py


# In[39]:


import sklearn.cluster as cluster
import sklearn.metrics as metrics


# In[40]:


df = pd.read_csv('C:/Users/abhis/OneDrive/Documents/o2_clusters.csv')
df.head()


# In[41]:


# input matrix for segmentation
val = df[['x','y','z']].values
print(val)


# In[42]:


for i in range(7,16):
    labels=cluster.KMeans(n_clusters=i,init="k-means++",random_state=200).fit(val).labels_
    print ("Silhouette score for k(clusters) = "+str(i)+" is "
           +str(metrics.silhouette_score(val,labels,metric="euclidean",sample_size=1000,random_state=200)))


# In[43]:


#Elbow method to find number of cluster
K=range(3,16)
wss = []
for k in K:
    kmeans=cluster.KMeans(n_clusters=k,init="k-means++")
    kmeans=kmeans.fit(val)
    wss_iter = kmeans.inertia_
    wss.append(wss_iter)


# In[44]:


mycenters = pd.DataFrame({'Clusters' : K, 'WSS' : wss})
mycenters


# In[45]:


sns.scatterplot(x = 'Clusters', y = 'WSS', data = mycenters, marker="+")
# We get 5 Clusters


# In[48]:


# find the optimal number of clusters using elbow method  
# find the optimal number of clusters using elbow method  
WCSS = []
for i in range(3,16):
    model = KMeans(n_clusters = i,init = 'k-means++')
    model.fit(x)
    WCSS.append(model.inertia_)
fig = plt.figure(figsize = (7,7))
plt.plot(range(3,16),WCSS, linewidth=4, markersize=12,marker='o',color = 'red')
plt.xticks(np.arange(11))
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# In[57]:


# finding the clusters based on input matrix "x"
model = KMeans(n_clusters = 10, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_clusters = model.fit_predict(x)


# In[58]:


# countplot to check the number of clusters and number of customers in each cluster
sns.countplot(y_clusters)


# In[59]:


print(x[y_clusters == 0,0][1])
print(x[y_clusters == 0,1][1])
print(x[y_clusters == 0,2][1])
print(x[y_clusters == 0,0][2])
print(x[y_clusters == 0,1][2])
print(x[y_clusters == 0,2][2])
print(x[y_clusters == 0,0][3])
print(x[y_clusters == 0,1][3])
print(x[y_clusters == 0,2][3])
print(x[y_clusters == 0,0][4])
print(x[y_clusters == 0,1][4])
print(x[y_clusters == 0,2][4])


# In[ ]:





# In[60]:


# 3d scatterplot using matplotlib

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[y_clusters == 0,0],x[y_clusters == 0,1],x[y_clusters == 0,2], s = 40 , color = 'blue', label = "cluster 0")
ax.scatter(x[y_clusters == 1,0],x[y_clusters == 1,1],x[y_clusters == 1,2], s = 40 , color = 'orange', label = "cluster 1")
ax.scatter(x[y_clusters == 2,0],x[y_clusters == 2,1],x[y_clusters == 2,2], s = 40 , color = 'green', label = "cluster 2")
ax.scatter(x[y_clusters == 3,0],x[y_clusters == 3,1],x[y_clusters == 3,2], s = 40 , color = 'red', label = "cluster 3")
ax.scatter(x[y_clusters == 4,0],x[y_clusters == 4,1],x[y_clusters == 4,2], s = 40 , color = 'yellow', label = "cluster 4")
ax.scatter(x[y_clusters == 5,0],x[y_clusters == 5,1],x[y_clusters == 5,2], s = 40 , color = 'black', label = "cluster 5")
ax.scatter(x[y_clusters == 6,0],x[y_clusters == 6,1],x[y_clusters == 6,2], s = 40 , color = 'pink', label = "cluster 6")
ax.scatter(x[y_clusters == 7,0],x[y_clusters == 7,1],x[y_clusters == 7,2], s = 40 , color = 'grey', label = "cluster 7")
ax.scatter(x[y_clusters == 8,0],x[y_clusters == 8,1],x[y_clusters == 8,2], s = 40 , color = 'violet', label = "cluster 8")
ax.scatter(x[y_clusters == 9,0],x[y_clusters == 9,1],x[y_clusters == 9,2], s = 40 , color = 'brown', label = "cluster 9")
ax.scatter(x[y_clusters == 10,0],x[y_clusters == 10,1],x[y_clusters == 10,2], s = 40 , color = 'black', label = "cluster 10")
ax.set_xlabel('x-->')
ax.set_ylabel('y-->')
ax.set_zlabel('z-->')
ax.legend()
plt.show()


# In[61]:


# 3d scatterplot using plotly
Scene = dict(xaxis = dict(title  = 'x -->'),yaxis = dict(title  = 'y--->'),zaxis = dict(title  = 'z-->'))

# model.labels_ is nothing but the predicted clusters i.e y_clusters
labels = model.labels_
trace = go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2], mode='markers',marker=dict(color = labels, size= 10, line=dict(color= 'black',width = 10)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 800,width = 800)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()


# In[ ]:




