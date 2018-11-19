

```python
import pandas as pd
import numpy as np
#etc

from sklearn import cluster

```


```python
iris = pd.read_csv("https://raw.githubusercontent.com/PyDataWorkshop/datasets/master/iris-skl.csv")
```


```python
iris.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
    Sepal Length    150 non-null float64
    Sepal Width     150 non-null float64
    Petal Length    150 non-null float64
    Petal Width     150 non-null float64
    Species         150 non-null object
    dtypes: float64(4), object(1)
    memory usage: 5.9+ KB


### Column Names


```python
iris.columns.tolist()
```




    ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species']




```python
iris.columns[iris.columns.str.startswith('Pet')]
```




    Index(['Petal Length', 'Petal Width'], dtype='object')




```python
iris.columns[iris.columns.str.endswith('dth')].tolist()
```




    ['Sepal Width', 'Petal Width']




```python
iris.columns = iris.columns.str.replace(' ','')
```


```python
iris.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SepalLength</th>
      <th>SepalWidth</th>
      <th>PetalLength</th>
      <th>PetalWidth</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris["Species"].head(5)
```




    0    setosa
    1    setosa
    2    setosa
    3    setosa
    4    setosa
    Name: Species, dtype: object




```python
iris["Species"].groupby(iris["Species"], axis=0).count()
```




    Species
    setosa        50
    versicolor    50
    virginica     50
    Name: Species, dtype: int64




```python
iris.columns[iris.columns.str.startswith('Sep')]
```




    Index(['SepalLength', 'SepalWidth'], dtype='object')




```python
iris.columns[iris.columns.str.endswith('gth')]
```




    Index(['SepalLength', 'PetalLength'], dtype='object')




```python
## df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
```


```python
### Remove White Space from Column Names
```


```python
#print(iris.columns.tolist())
iris.columns.str.replace('',' ') 
#iris.columns.tolist()

```




    Index([' S e p a l L e n g t h ', ' S e p a l W i d t h ',
           ' P e t a l L e n g t h ', ' P e t a l W i d t h ', ' S p e c i e s '],
          dtype='object')




```python
iris.columns.str.replace(' ','') 

```




    Index(['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'], dtype='object')




```python
iris.columns = iris.columns.str.replace(' ','')
```


```python
iris[1:4] #Rows
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SepalLength</th>
      <th>SepalWidth</th>
      <th>PetalLength</th>
      <th>PetalWidth</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



*** Not what we want! Python is "Zero Index" ***

### integer location


```python
iris.iloc[:, [1]].head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SepalWidth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris.iloc[:,:4].head(5)

# All Rows
# First 4 Columns
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SepalLength</th>
      <th>SepalWidth</th>
      <th>PetalLength</th>
      <th>PetalWidth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris_feature = iris.iloc[:,:4]
iris_feature.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SepalLength</th>
      <th>SepalWidth</th>
      <th>PetalLength</th>
      <th>PetalWidth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris_targ = iris["Species"]
iris_targ.head(6)
```




    0    setosa
    1    setosa
    2    setosa
    3    setosa
    4    setosa
    5    setosa
    Name: Species, dtype: object




```python
# alternative approach
```


```python
iris.drop(iris.columns[:4], axis=1).head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
type(iris_targ)
```




    pandas.core.series.Series




```python
k = 3
kmeans = cluster.KMeans(n_clusters=k)

```


```python
kmeans.fit(iris_feature)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=3, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0)




```python
# to get the locations of the centroids and the label of the owning cluster for each observation in the data set:

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
```


```python
labels
```




    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2,
           0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
           2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2], dtype=int32)




```python
pd.crosstab(iris_targ,labels)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
    <tr>
      <th>Species</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>setosa</th>
      <td>0</td>
      <td>50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>versicolor</th>
      <td>2</td>
      <td>0</td>
      <td>48</td>
    </tr>
    <tr>
      <th>virginica</th>
      <td>36</td>
      <td>0</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>




```python
centroids,labels,inertia = cluster.k_means(iris_feature,n_clusters=k,n_init=25,algorithm = "elkan")
```


```python
labels
```


```python
type(pd.crosstab(iris_targ,labels))
```


```python
df = pd.crosstab(iris_targ,labels)
column_titles = [1,2,0]

df.reindex(columns=column_titles)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>1</th>
      <th>2</th>
      <th>0</th>
    </tr>
    <tr>
      <th>Species</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>setosa</th>
      <td>50</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>versicolor</th>
      <td>0</td>
      <td>48</td>
      <td>2</td>
    </tr>
    <tr>
      <th>virginica</th>
      <td>0</td>
      <td>14</td>
      <td>36</td>
    </tr>
  </tbody>
</table>
</div>


