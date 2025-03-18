# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import AgglomerativeClustering

# Load data
dataset = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

# Performing exploratory data analysis (EDA)

## Checking number of rows and columns
print('Number of samples and features:')
print(dataset.shape, '\n\n')

## Checking features data types
print('Data types and its numbers:')
print(dataset.dtypes.value_counts(ascending=False), '\n\n')

## Analyzing the responses for categorical features
print('Viewing responses from categorical features:')
for feat in dataset.select_dtypes(include='object').columns:
    print(dataset[feat].value_counts(ascending=False, normalize=True, dropna=False)*100, '\n\n')

print('Viewing number of unique values for features with multiple different values:')
print("For 'Why or why not?' feature")
print(dataset['Why or why not?'].nunique(), '\n\n')
print("For 'Why or why not?.1' feature:")
print(dataset['Why or why not?.1'].nunique(), '\n\n')

## Analyzing the responses for numerical features
print('Viewing responses from numerical features:')
for feat in dataset.select_dtypes(exclude='object').columns:
    print(dataset[feat].value_counts(ascending=False, dropna=False), '\n\n')

## Checking for null values
print('Proportion of null values in dataset:')
print(dataset.isnull().sum().sort_values(ascending=False).values*100 / len(dataset), '\n\n')

# Data Preprocessing

## Dropping features and rows
dataset = dataset.drop(columns=['Why or why not?',
                                'Why or why not?.1'])

## Removing features with more than 21% of missing values
selected_feats = []
for feat in dataset.columns:
    prop_null_values = dataset[feat].isnull().sum() / len(dataset)
    if prop_null_values < 0.6:
        selected_feats.append(feat)
dataset = dataset[selected_feats]

## Rechecking the proportion of null values
print('Proportion of null values after excluding features with too much null values:')
print(dataset.isnull().sum().sort_values(ascending=False).values*100 / len(dataset), '\n\n')

## Rechecking features data types
print('Data types and its numbers:')
print(dataset.dtypes.value_counts(ascending=False), '\n\n')

## Checking for numerical features missing values
print('Numerical features with missing values:')
for feat in dataset.select_dtypes(exclude='object').columns:
    if dataset[feat].isnull().sum() != 0:
        print(dataset[feat].value_counts(ascending=False, dropna=False), '\n\n')

## Modifying values
male = ['Male', 'male', 'Male ', 'M', 'm', 'man', 'male 9:1 female, roughly','Male (cis)', 'Cis male', 'Male.', 'Man',
        'Sex is male', 'cis male', 'Malr', 'Dude',
        "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ",
        'mail', 'M|', 'Male/genderqueer', 'male ', 'Cis Male', 'Male (trans, FtM)', 'cisdude', 'cis man', 'MALE']
female = ['Female', 'female', 'I identify as female.', 'female ', 'Cis female ', 'Transitioned, M2F',
          'Genderfluid (born female)', 'Female or Multi-Gender Femme', 'Female ', 'woman', 'female/woman',
          'Cisgender Female', 'fem', 'Female (props for making this a freeform field, though)', ' Female', 'Cis-woman',
          'female-bodied; no feelings about gender', 'AFAB', 'F', 'f', 'Woman', 'fm', 'Female assigned at birth ']
other_genders = ['Bigender', 'non-binary', 'Other/Transfeminine',  'Androgynous', 'Other', 'nb masculine',
                'none of your business', 'genderqueer', 'Human', 'Genderfluid', 'Enby', 'genderqueer woman', 'mtf',
                'Queer','Agender', 'Fluid', 'Nonbinary', 'human', 'Unicorn', 'Genderqueer', 'Genderflux demi-girl',
                'Transgender woman']

dataset["What is your gender?"] = dataset["What is your gender?"].replace(to_replace=male, value='male')
dataset["What is your gender?"] = dataset["What is your gender?"].replace(to_replace=female, value='female')
dataset["What is your gender?"] = dataset["What is your gender?"].replace(to_replace=other_genders, value='other genders')

## Fill missing values for numerical features
dataset = dataset.fillna({'Is your employer primarily a tech company/organization?':1.0})
print('Checking for missing values in the numerical features:')
print(dataset['Is your employer primarily a tech company/organization?']\
      .value_counts(ascending=False, dropna=False), '\n\n')

## Filling missing values for categorical features
dataset = dataset.fillna(value='Missing')
print('Proportion of null values after filling missing values:')
print(dataset.isnull().sum().sort_values(ascending=False).values*100 / len(dataset), '\n\n')

## Excluding outliers
print("Looking for outliers inside feature 'What is your age?':")
print(np.sort(dataset['What is your age?'].unique()), '\n\n')
dataset = dataset[(dataset['What is your age?'] != 3) & (dataset['What is your age?'] != 323)]
print('Checking if outliers were correctly excluded:')
print(np.sort(dataset['What is your age?'].unique()), '\n\n')

## Encoding categorical variables and standardizing values
ohe = OneHotEncoder(sparse_output=False)
scaler = StandardScaler()
transformer = ColumnTransformer([('cat_cols', ohe, dataset.select_dtypes(include='object').columns),
                                 ('num_cols', scaler, dataset.select_dtypes(exclude='object').columns)])
transformer.fit(dataset)
preprocessed_dataset = pd.DataFrame(data=transformer.transform(dataset), columns=transformer.get_feature_names_out())
print('Number of samples and features after preprocessing dataset:')
print(preprocessed_dataset.shape, '\n\n')

## Applying dimensionality reduction
pca = PCA(n_components=3)
pca.fit(preprocessed_dataset)

dataset_pca = pd.DataFrame(data=pca.transform(preprocessed_dataset), columns=['PC1', 'PC2', 'PC3'])

# Labeling the clusters
agg = AgglomerativeClustering(n_clusters=8, linkage='average')
labels = agg.fit_predict(dataset_pca)
dataset_pca['labels'] = labels

# Discovering what each cluster means

## Discovering mean values for each cluster in each PC
cluster_pca_summary = dataset_pca.groupby('labels').mean()
print('Clusters means:')
print(cluster_pca_summary, '\n\n')

## Discovering the most relevant features for the PC
loadings = pd.DataFrame(data=pca.components_.T,
                        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                        index=preprocessed_dataset.columns).sort_values(ascending=False, by=['PC2'])
relevant_pc1_pos_feats = loadings[loadings['PC1'] > 0.15]
relevant_pc1_neg_feats = loadings[loadings['PC1'] < -0.15]
relevant_pc2_pos_feats = loadings[loadings['PC2'] > 0.15]
relevant_pc2_neg_feats = loadings[loadings['PC2'] < -0.15]
relevant_pc3_pos_feats = loadings[loadings['PC3'] > 0.15]
relevant_pc3_neg_feats = loadings[loadings['PC3'] < -0.15]

print('Relevant PC1 positive features:')
print(relevant_pc1_pos_feats['PC1'], '\n\n')
print('Relevant PC1 negative features:')
print(relevant_pc1_neg_feats['PC1'], '\n\n')
print('Relevant PC2 positive features:')
print(relevant_pc2_pos_feats['PC2'], '\n\n')
print('Relevant PC2 negative features:')
print(relevant_pc2_neg_feats['PC2'], '\n\n')
print('Relevant PC3 positive features:')
print(relevant_pc3_pos_feats['PC3'], '\n\n')
print('Relevant PC3 negative features:')
print(relevant_pc3_neg_feats['PC3'], '\n\n')


## Analyzing the most relevant features values
feats = []
most_relevant_features = relevant_pc1_pos_feats + relevant_pc1_neg_feats \
                         + relevant_pc2_pos_feats + relevant_pc2_neg_feats \
                         + relevant_pc3_pos_feats + relevant_pc3_neg_feats
for feature in most_relevant_features.index:
    strings = feature.split('_')
    category = strings[0]
    feature = strings[-1] if category == 'num' else strings[-2]
    if category == 'num' and feature not in feats:
        feats.append(feature)
    if category == 'cat' and feature != '' and feature not in feats:
        feats.append(feature)

## Analyzing the characteristics inside a cluster
print('Analyzing cluster characteristics through the most relevant features:')
selected_indexes = [4,1,5,0,6,3,7,2]
for feat in feats:
    for i in selected_indexes:
        cluster = dataset[labels == i][feat].value_counts(normalize=True)*100
        print(f'For Cluster {i}')
        print(cluster, '\n\n')

# Visualizing dataset
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
p = ax.scatter(xs=dataset_pca['PC1'], ys=dataset_pca['PC2'], zs=dataset_pca['PC3'],
               c=labels, cmap='viridis', alpha=0.15)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.colorbar(p, label='Cluster Labels', ticks=[0,1,2,3,4,5,6,7])

def analyze_sample(label, show_sample=False):
    sample = dataset[labels == label]
    sample = transformer.transform(sample)
    sample = pca.transform(sample)
    if show_sample:
        ax.scatter(xs=sample[:,0], ys=sample[:,1], zs=sample[:,2], color='red', s=100, marker='x', alpha=1)

analyze_sample(label=4, show_sample=False)
plt.show()