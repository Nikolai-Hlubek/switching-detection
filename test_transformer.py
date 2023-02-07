#!/usr/bin/env python
# coding: utf-8

# <span style='font-weight:bold; font-size:xx-large'>Machine learning classification</span>
# 
# **Author:** *Nikolai Hlubek*
# 
# Try to predict the switching using machine learning. 

# In[1]:


import os
import pickle

import numpy as np
import pandas as pd

# Static plots
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib

# Dynamic plots
import plotly.express as px

# Filtering
import scipy.signal

# Jupyter display infrastructure
import IPython.display

# Machine learning library
import sklearn
import sklearn.model_selection
import sklearn.cross_decomposition
import sklearn.cluster
import sklearn.ensemble
import sklearn.svm

# Neuronal networks
import keras


# # Parameters

# In[2]:


filename_out = 'models#KeilE_success#Transformer.pkl'


# # Load data

# In[3]:


with open('df_current.pkl', 'rb') as file:
    df_current = pickle.load(file)

with open('df_voltage.pkl', 'rb') as file:
    df_voltage = pickle.load(file)

with open('df_params.pkl', 'rb') as file:
    df_params = pickle.load(file)


# In[4]:


X = df_current.copy()


# In[5]:


y_bool = df_params['fault']


# # Smooth data

# In[6]:


X_savgol = X.apply(lambda x: scipy.signal.savgol_filter(x.values, 5, 2, mode='nearest'), axis=1, result_type='broadcast')


# In[7]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

idx = 150
ax1.plot(X.loc[idx,:], linewidth=3)
ax1.plot(X_savgol.loc[idx,:], linewidth=1)

ax2.plot(X.loc[idx,:], linewidth=3)
ax2.plot(X_savgol.loc[idx,:], linewidth=1)
ax2.set_ylim(42,45)


# # Scaling

# In[8]:


X_max = X.div(X.max(axis=1), axis=0)


# In[9]:


X_snv = sklearn.preprocessing.scale(X, with_mean=True, with_std=True)


# In[10]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))

ax1.plot(X.loc[2,:])
ax2.plot(X_max.loc[2,:])
ax3.plot(X.columns, X_snv[2,:])


# # To categorical

# In[11]:


y = y_bool.to_numpy()
y = y.astype('int')


# # Metrics

# In[12]:


def calc_metrics(y_truth, y_predict, print=False):
    # Calculate scores
    r2_score = sklearn.metrics.r2_score(y_truth, y_predict)
    ev_score = sklearn.metrics.explained_variance_score(y_truth, y_predict)
    mse = sklearn.metrics.mean_squared_error(y_truth, y_predict)
    acc_score = sklearn.metrics.accuracy_score(y_truth, y_predict)
    precision_score_None = sklearn.metrics.precision_score(y_truth, y_predict)
    recall_score_None = sklearn.metrics.recall_score(y_truth, y_predict)
    precision_score_macro = sklearn.metrics.precision_score(y_truth, y_predict, average='macro')
    recall_score_macro= sklearn.metrics.recall_score(y_truth, y_predict, average='macro')
    misclassifications = sklearn.metrics.zero_one_loss(y_truth, y_predict, normalize=False)

    if print:
        print("R2 Score: {:.4f}".format(r2_score))
        print("Explained variance score: {:.4f}".format(ev_score))
        print("Mean squared error: {:.4f}".format(mse))
        print("Total number of misclassifications: {}".format(misclassifications))
        print("Accuracy Score: {:.4f}".format(acc_score))
        print(f"Precision: {np.array2string(precision_score_None, precision=4, floatmode='fixed')}")
        print(f"Recall: {np.array2string(recall_score_None, precision=4, floatmode='fixed')}")
        print(f"Precision macro: {precision_score_macro}")
        print(f"Recall macro: {recall_score_macro}")

    return {'Misclassifications': misclassifications, 'r2 score': r2_score, 'Explained variance score': ev_score, 'Accuracy score': acc_score, 'Precision': precision_score_None, 'Recall': recall_score_None}


# In[13]:


def plot_quality(y_truth, y_predict):
    fig, ax = plt.subplots()
    ax.scatter(y_truth, y_predict, edgecolors=(0, 0, 0))
    ax.plot([y_truth.min(), y_truth.max()], [y_truth.min(), y_truth.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    
    return fig, ax


# In[14]:


def plot_misclassifications(y_cv, y=y):
    failures_idx = []
    for i in range(len(y)):
        if y[i] != y_cv[i]:
            failures_idx.append(i)

    fig, ax = plt.subplots()

    for idx in failures_idx:
        ax.plot(X.loc[idx])

    ax.set_title('Wrongly classified')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('current (mA)')

    IPython.display.display(df_params.loc[failures_idx])


# # Model storage

# In[15]:


model_overview = []


# # Machine learning algorithms

# In[16]:


def run_multiple(func, X, y, method, model_overview, n=10):
    """
    Run the model a few times to get the best solution
    """
    for i in range(n):
        y_cv = func(X, y, method, model_overview)
        quality = model_overview[-1]
        if i == 0:
            quality_best = quality.copy()
            y_cv_best = y_cv.copy()
        if quality['Misclassifications'] < quality_best['Misclassifications']:        
            quality_best = quality.copy()
            y_cv_best = y_cv.copy()
    
    return y_cv_best


# ## Transformer

# According to  
# https://keras.io/examples/timeseries/timeseries_transformer_classification/

# In[17]:


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = keras.layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def calc_1D_transformer(X, y, method, model_overview):
    # One hot encoding for keras
#    y_ohe = sklearn.preprocessing.OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    
    #get number of columns in training data
    n_timesteps = X.shape[1]
    n_featues = X.shape[2]
    
    head_size = 256 #256
    num_heads = 4
    num_transformer_blocks = 4
    ff_dim = 4
    dropout = 0.25
    mlp_units = [128] # 128
    mlp_dropout = 0.4
    
    n_classes = len(np.unique(y))
    
    inputs = keras.Input(shape=(n_timesteps, n_features))
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)
    for dim in mlp_units:
        x = keras.layers.Dense(dim, activation='relu')(x)
        x = keras.layers.Dropout(mlp_dropout)(x)
#    outputs = keras.layers.Dense(n_classes, activation='softmax')(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam',
        metrics=['binary_accuracy'],
#        loss='sparse_categorical_crossentropy', 
#        optimizer='adam',
#        metrics=['sparse_categorical_accuracy'],
    )
    
#    print(model.summary())

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    #model.fit(X, y_ohe, validation_split=0.2, epochs=200, batch_size=64, callbacks=callbacks)
    model.fit(X, y, validation_split=0.2, epochs=200, batch_size=64, callbacks=callbacks)

    y_cv_probability = model.predict(X)
    
    # Transform probablity result to classification using a threshold
    # https://visualstudiomagazine.com/articles/2018/08/30/neural-binary-classification-keras.aspx 
    # Single sigmoid output node holds a value between 0 and 1 that holds the probablity 
    # of the output being in class 1 or not. 
    y_cv_classification = np.array(y_cv_probability).flatten() > 0.1
    y_cv_classification = y_cv_classification.astype(int)
    
    # Inverse one hot encoding
#    y_cv = []
#    for i in range(len(y_cv_ohe)):
#        if y_cv_ohe[i][0] > y_cv_ohe[i][1]:
#            y_cv.append(1)
#        else:
#            y_cv.append(0)

    metrics = calc_metrics(y, y_cv_classification)
    metrics['method'] = method
    metrics['cv'] = y_cv_classification
    metrics['model'] = model
    model_overview.append(metrics)
    # Store intermediate results to file because training takes a long time
    with open(filename_out, 'ab+') as fp:
        pickle.dump(metrics, fp)


    #plot_quality(y, y_cv)

    return y_cv_classification


# In[18]:


X_red_raw = X.loc[:,0:7.5].copy()


# In[19]:


X_red_max = X_red_raw.div(X_red_raw.max(axis=1), axis=0)
X_red_snv = sklearn.preprocessing.scale(X_red_raw, with_mean=True, with_std=True)


# In[20]:


n_features = 1 # Anzahl der Sensoren
X_raw_keras = X_red_raw.values.reshape((X_red_raw.shape[0], X_red_raw.shape[1], n_features))
X_max_keras = X_red_max.values.reshape((X_red_max.shape[0], X_red_max.shape[1], n_features))
X_snv_keras = X_red_snv.reshape((X_red_snv.shape[0], X_red_snv.shape[1], n_features))


# In[21]:


y_cv_raw = run_multiple(calc_1D_transformer, X_raw_keras, y, '1D transformer raw', model_overview, n=5)
y_cv_max = run_multiple(calc_1D_transformer, X_max_keras, y, '1D transformer max', model_overview, n=5)
y_cv_snv = run_multiple(calc_1D_transformer, X_snv_keras, y, '1D transformer snv', model_overview, n=5)


# In[22]:


pd.DataFrame(model_overview)


# In[23]:


plot_misclassifications(y_cv_max)


# In[24]:


plot_misclassifications(y_cv_snv)


# In[25]:


pd.DataFrame(model_overview)


# In[ ]:




