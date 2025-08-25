import pandas as pd

def generateAI():
    dataset=pd.read_csv('ml-data.csv')
    dataset=dataset.dropna()                       #delete NaN rows              
    X=dataset.iloc[:,1].values                     #sepearte X values(i.e. temp) from dataset
    X=X.reshape(-1,1)                              #converts 1D array to 2D array
    y=dataset.iloc[:,-1].values                    #load y values
    from sklearn.preprocessing import LabelEncoder # Encode the target variable
    le = LabelEncoder()
    y = le.fit_transform(y)

    from sklearn.model_selection import train_test_split  #split the data
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

    from sklearn.neighbors import KNeighborsRegressor  # applying k nearest neighbors (utility distance)
    ai=KNeighborsRegressor(n_neighbors=5)         #here no. of neighbours=5
    ai.fit(X_train,y_train)
    import pickle
    pickle.dump(ai,open('model.pkl','wb'))