import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt



def get_ensemble_models():
    rf = RandomForestRegressor(n_estimators=150,min_samples_leaf=1,min_samples_split=2,random_state=42)
    grad = GradientBoostingRegressor(n_estimators=200, min_samples_leaf=1,min_samples_split=3, random_state=42)
    extra = ExtraTreesRegressor(n_estimators=150, random_state=42)
    classifier_list = [rf, grad, extra]
    classifier_name_list = ['Random Forest','Gradient Boosting','Extra Trees']
    return classifier_list, classifier_name_list
    
    
    
    def print_evaluation_metrics(trained_model,trained_model_name,X_test,y_test):
    print ('--------- For Model : ', trained_model_name ,' ---------\n')
    predicted_values = trained_model.predict(X_test)
    #print ("Predicted Values : ", predicted_values)
    print ("Mean Absolute Error : ", metrics.mean_absolute_error(y_test,predicted_values))
    print ("Mean Squared Error : ", metrics.mean_squared_error(y_test,predicted_values))
    print ("R2 Score : ", metrics.r2_score(y_test,predicted_values))
    print ("---------------------------------------\n")
    actual_values = y_test[30000:30050]
    plt.plot(predicted_values[30000:30050], color='green', label='Predicted Values')
    plt.plot(actual_values, color='red', label='Actual Values')
    plt.xlabel('Week Number')
    plt.ylabel('Weekly Sales')
    plt.legend(loc='upper right')
    plt.title('Comparison of Predicted and Actual Weekly Sales for ', trained_model_name)
    plt.show()
    
    
    
    
    def label_encode_frame(dataframe):
    columns = dataframe.columns
    encoder = LabelEncoder()
    for column in columns:
        if type(dataframe[column][0]) is np.nan:
            for i in range(len(dataframe)):
                if i > 1000:
                    break
                if type(dataframe[column][i]) is str or type(dataframe[column][i]) is bool:
                    dataframe[column] = encoder.fit_transform(dataframe[column].values)
                    break
        elif type(dataframe[column][0]) is str or type(dataframe[column][0]) is bool:
            dataframe[column] = encoder.fit_transform(dataframe[column].values)
    return dataframe
    
    
    
    
    
    
train_frame = pd.read_csv(r"C:\Users\Nikhil\Desktop\train.csv")
columns_to_delete = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','Date']
train_frame.drop(columns_to_delete,axis=1,inplace=True)
target_values = list(train_frame['Weekly_Sales'].values)
del train_frame['Weekly_Sales']
train_frame = label_encode_frame(train_frame)
X_train,X_test,y_train,y_test = train_test_split(train_frame.values,target_values,test_size=0.2,random_state=42)
regressor_list,regressor_name_list = get_ensemble_models()
for regressor,regressor_name in zip(regressor_list,regressor_name_list):
    regressor.fit(X_train,y_train)
    print_evaluation_metrics(regressor,regressor_name,X_test,y_test)
