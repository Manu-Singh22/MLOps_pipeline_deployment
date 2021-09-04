#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import json
import numpy as np


from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from azureml.core.model import Model


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    """
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "sklearn_regression_model.pkl"
    )
    # deserialize the model file back into a sklearn model
    
    """
    model_path=Model.get_model_path('RTO-pipeline')
    print(model_path)
    model = load_model(model_path)
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    
    """
    #model = load_model('RTO_model4_regularizer.h5')
    logging.info("Request received")
    data = json.loads(raw_data)["data"]
    #data = pd.DataFrame.from_dict(raw_data)
    scaler=MinMaxScaler()
    data=np.array(data)
    X=scaler.fit_transform(data)
    X_re=X.reshape(X.shape[0],1,X.shape[1])
    R = model.predict(X_re)
    R_re=R.reshape(R.shape[0],R.shape[2])
    result=scaler.inverse_transform(R_re)
    logging.info("Request processed")
    return result.tolist()


# In[ ]:





