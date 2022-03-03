import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns                       
import sys
import xlrd 
import warnings                                  
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, accuracy_score
from sklearn.metrics import mean_squared_error
#------------------------------------------------------------------------------------------------------------------------------------
st.header('Automating Some Task of Data Scientist !!!')
add_selectbox = st.sidebar.markdown(':sunglasses: Name:Prathamesh Laxman Kashid :sunglasses:')
add_selectbox = st.sidebar.markdown("------------------------------")
add_selectbox = st.sidebar.markdown("Future scope: To integrate DL models :sunglasses:")
add_selectbox = st.sidebar.markdown("------------------------------")

#-----------------------------------------------------------------------------------------------------------------------------------
st.error("Hi Prathamesh Here...!!! May I Known Your Good Name ???")
name = st.text_input("User")
if name:
    n = ("Welcome \t" + name + '\tglad to see you. Please upload for the analysis'+'\t!!!')
    st.warning(n)
    uploaded_file = st.file_uploader("Upload:",type=['csv','xlsx'])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!!! File type is CSV !!!")
        except Exception:
            df = pd.read_excel(uploaded_file)
            df=pd.DataFrame(df)
            st.success("File uploaded successfully!!! File type is Excel !!!")
#----------------------------------------------------------------------------------------------
        try:
            st.write("Dataframe:")
            st.dataframe(df)
            st.success("Data as Readed successfully!!!")
        except Exception:
            st.error("Can't Fetch For Moment")
#-----------------------------------------------------------------------------------------
        c=df.info
        st.write('Info of Dataframe:',c)
        st.success('Info of Dataframe is displayed successfully!!!')
#-----------------------------------------------------------------------------------------
        e=df.isnull().sum().sort_values(ascending=False)
        data = {'Null_Values': e }
        data=pd.DataFrame(data)
        st.write('Checking null values in Dataframe:',data)
        st.success('Null values are displayed successfully!!!')
#------------------------------------------------------------------------------------------------
        f=(df.isnull().sum()*100/df.isnull().count()).sort_values(ascending=False)
        data = {'Null_Values %': f}
        data=pd.DataFrame(data)
        st.write('Percentage of null values in Dataframe:',data)
        st.success(' Null values % are displayed successfully!!!')
#-----------------------------------------------------------------------------------------------     
        if df.isnull().values.any():
            df.dropna( axis=0, inplace=True)
            df1=df.isnull().sum().sort_values(ascending=False)
            data = {'Null_Values': df1 }
            data=pd.DataFrame(data)
            st.write('Checking null values in Dataframe:',data)
            st.success("Successfully droped all null values")
#--------------------------------------------------------------------------------------------------------------------------------------
        st.write('Shape of Dataframe:')
        try:
            a=df.shape
            st.write(a)
            st.success('Shape of dataframe is displayed successfully!!!')
            if a[0]==0:
                st.datatframe(df)
                st.error("Bad file for analysis contain no data after dropping null values")
        except Exception:
            st.write('Shape of Dataframe',len(df))
            st.success('Shape of dataframe is displayed successfully!!!')
            
#--------------------------------------------------------------------------------------------------------------------------------
        d=df.columns
        data = {'column_names': d }
        data=pd.DataFrame(data)
        st.write('Columns in Datafrane:',data)
        st.success('Columns of dataframe are displayed successfully!!!')
#--------------------------------------------------------------------------------------------------------------------------------
        st.write("Numerical columns in Datafrane:")
        numeric_data = df.select_dtypes(include=[np.number])
        st.write(numeric_data)
        st.success('Numerical columns in dataframe are displayed successfully!!!')
#--------------------------------------------------------------------------------------------------------------------------------
        st.write("Categorical columns in Datafrane:")
        categorical_data = df.select_dtypes(exclude=[np.number])
        try:
            st.write(categorical_data)
            st.success('categorical columns in dataframe are displayed successfully!!!')
        except Exception:
            st.warning("No categorical Column !!!")
#-------------------------------------------------------------------------------------------------------------------------------
        st.write('Describing Datafrane:')
        b=df.describe()
        st.dataframe(b)
        st.success('Described Dataframe successfully!!!')
#--------------------------------------------------------------------------------------------------------------------------------
        st.write('Correlation of columns in Dataframe:')
        d=df.corr()
        st.dataframe(d)
        st.success('Correlation of Dataframe are displayed successfully!!!')
#--------------------------------------------------------------------------------------------------------------------------------
        st.write('Outlier:')
        numeric_data = df.select_dtypes(include=[np.number])
        st.warning("checking for outlier!!!")
        for i in numeric_data:
            fig=plt.figure(figsize = (10,10))
            sns.boxplot(x=df[i])
            plt.show()
            st.pyplot(fig)
        try:
            Q1 = np.percentile(df[i], 25,interpolation = 'midpoint')
            Q3 = np.percentile(df[i], 75,interpolation = 'midpoint')
            IQR = Q3 - Q1
            a=df.shape
            upper = np.where(df[i] >= (Q3+1.5*IQR))
            lower = np.where(df[i] <= (Q1-1.5*IQR))
            df.drop(upper[0], inplace = True)
            df.drop(lower[0], inplace = True)
            b=df.shape
            if a==b:
                st.warning('No Outlier Found')
            else:
                st.error(" OUTLIER IS FOUND!!!")
                st.warning('Removing outlier')
                st.write("Old Shape: ", a)
                st.write("New Shape: ", b)
                st.success(" OUTLIER IS REMOVED SUCCESSFULLY!!!")
        except Exception:
            st.error("Removing Outlier Action Can't Preform for a moment")
#----------------------------------------------------------------------------------------------------------------------------
        st.write("Histogram:")
        numeric_data = df.select_dtypes(include=[np.number])
        fig=plt.figure(figsize=(30,20))
        plt.hist(numeric_data)
        st.pyplot(fig)
        st.success('Histogram plotted successfully!!!')
#-------------------------------------------------------------------------------------------------------------------------------
        st.write('Correlation Plot:')
        fig=sns.set(rc={"figure.figsize":(10, 10)})
        sns.heatmap(df.corr(),annot=True,cmap='coolwarm');
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)
        st.success('Correlation plot plotted successfully!!!')
#-------------------------------------------------------------------------------------------------------------------------------
        st.write('Heat map:')
        numeric_data = df.select_dtypes(include=[np.number])
        fig=plt.figure(figsize=(15,7))
        sns.heatmap(numeric_data);
        st.pyplot(fig)
        st.success('Heatmap plotted successfully!!!')
#-------------------------------------------------------------------------------------------------------------------------------
        st.write('Count plot:')
        for i in df:
            fig=plt.figure(figsize = (10,10))
            if(df[i].nunique()<30):
                sns.countplot(x=df[i])
                plt.show()
                st.pyplot(fig)
        st.success('Countplot plotted successfully!!!')
#-------------------------------------------------------------------------------------------------------------------------------
        st.write('Pair Plot:')
        fig=sns.pairplot(df)
        st.pyplot(fig)
        st.success('Pairplot plotted successfully!!!')
#---------------------------------------------------------------------------------------------------------------------------------
        st.write("Select Columns for X & Y:")
#-------------------------------------------------------------------------------------------------------------------------------
        col=df.columns
        x=st.multiselect("select columns for X:",col)
#--------------------------------------------------------------------------------------------------------------------------------
        if x:
            row=len(df[x])
            st.write("You have selected:",x)
            st.success("X were seletced sucessfully!!!")
            D1=pd.DataFrame(df[x],index=pd.RangeIndex(start=0, stop=row, step=1))
            D1.dropna( axis=0, inplace=True)
            st.dataframe(D1)            
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
        y=st.multiselect("select columns for Y:",col)
        if y:
            row1=len(df[y])
            st.write("You have selected:",y)
            st.dataframe(df[y])
            st.success("Y were seletced sucessfully!!!")
            D2=pd.DataFrame(df[y],index=pd.RangeIndex(start=0, stop=row1, step=1))
            D2.dropna( axis=0, inplace=True)
            st.write("Dataset of of your selected column:")
            data=D1.join(D2)   
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
            st.write("3.Scaling X:")
            if any(D1.select_dtypes(include=[np.number])):
                numeric_data = D1.select_dtypes(include=[np.number])
                st.write("Numeric Before Scaling:")
                st.dataframe(numeric_data)
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                st.write("Categorical After Scaling")
                numeric_data = sc.fit_transform(numeric_data)
                st.dataframe(numeric_data)
                n1=pd.DataFrame(numeric_data) #,index=pd.RangeIndex(start=0, stop=row2, step=1))
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
            if any(D1.select_dtypes(exclude=[np.number])):
                categorial_data = D1.select_dtypes(exclude=[np.number])
                st.write('Before One Hot Encding:')
                st.dataframe(categorial_data)
                categorial_data = pd.get_dummies(categorial_data)
                st.write('After One Hot Encding:')
                st.dataframe(categorial_data)
                n2=pd.DataFrame(categorial_data) #,index=pd.RangeIndex(start=0, stop=row2, step=1))
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            st.write("After scaling X:")
            try:
                training_feature=n1.join(n2)
                #training_feature=training_feature.dropna( axis=0, inplace=True)
                st.dataframe(training_feature)
            except Exception:
                training_feature=numeric_data
                #training_feature=training_feature.dropna( axis=0, inplace=True)
                st.dataframe(training_feature)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
            st.write("Train & Test Split:")
            X1=training_feature
            row9=len(X1)
            y1=df[y].head(row9)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.2,random_state=100)
            st.write("X_train:")
            #X_train=X_train.dropna( axis=0, inplace=True)
            st.dataframe(X_train)
            row2=len(X_train)
            st.write(row2)
            st.write("X_test:")
            st.dataframe(X_test)
            row3=len(X_test)
            st.write(row3)
            st.write("y_train:")
            st.dataframe(y_train)
            st.write("Y_test:")
            st.dataframe(y_test)        
#======================================================================================================================================================================
            st.write("Select ML Type:")
            click=st.checkbox('Supervised:')
            if click==True:
                st.write("Select ML Model:")
                b=('Linear Regression','Logistic Regression','Decision Tree','Naive Bayes','support vector machine','K-nearest Neighbor')
                click= st.selectbox("Select Model Type:",b)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                if click=='Linear Regression':
                    try:
                        st.write("Linear Regression Model:")
                        from sklearn.linear_model import LinearRegression
                        regressor= LinearRegression()
                        st.write(regressor)
                        regressor.fit(X_train,y_train)
                        st.success("Model created & fitted successfully!!!")
                        a=regressor.score(X_train,y_train)*100
                        y_predlog=regressor.predict(X_test)
                        st.write("Y_pred:",y_predlog)
                        st.success("Model has predictted successfully!!!")
                        acrr=accuracy_score(y_test,y_predlog)*100
                        st.write("Accuracy:",a)
                        st.success("Accuracy calculated successfully!!!")
                    except Exception:
                        st.error("This model is not suitable for your analysis try another!!!")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------s
                if click=='Logistic Regression':
                    try:
                        st.write("Logistic Regression Model:")
                        from sklearn.linear_model import LogisticRegression
                        log=LogisticRegression()
                        st.write(log)
                        log.fit(X_train,y_train)
                        st.success("Model created & fitted successfully!!!")
                        a=log.score(X_train,y_train)*100
                        y_predlog=log.predict(X_test)
                        st.write("Y_pred:",y_predlog)
                        st.success("Model has predictted successfully!!!")
                        acrr=accuracy_score(y_test,y_predlog)*100
                        st.write("Accuracy:",a)
                        st.success("Accuracy calculated successfully!!!")
                    except Exception:
                        st.error("This model is not suitable for your analysis try another!!!")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------s
                if click=='Decision Tree':
                    try:
                        st.write("Decision Tree Model:")
                        DT=DecisionTreeClassifier(criterion = "entropy")
                        DT.fit(X_train,y_train)
                        st.write(DT)
                        st.success("Model created &  fitted sucessfully!!!")
                        a=DT.score(X_train,y_train)*100
                        y_predlog=DT.predict(X_test)
                        st.write("Y_pred:",y_predlog)
                        st.success("Model has predictted successfully!!!")
                        acrr=accuracy_score(y_test,y_predlog)*100
                        st.write("Accuracy:",a)
                        st.success("Accuracy calculated successfully!!!")
                    except Exception:
                        st.error("This model is not suitable for your analysis try another!!!")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------s
                if click=='Naive Bayes':
                    try:
                        st.write("Naive Bayes Model:")
                        model=GaussianNB()
                        st.write(model)
                        model.fit(X_train,y_train)
                        st.success("Model created & fitted successfully!!!")
                        a=model.score(X_train,y_train)*100
                        y_predlog=model.predict(X_test)
                        st.write("Y_pred:",y_predlog)
                        st.success("Model has predictted successfully!!!")
                        acrr=accuracy_score(y_test,y_predlog)*100
                        st.write("Accuracy:",a)
                        st.success("Accuracy calculated successfully!!!")
                    except Exception:
                        st.error("This model is not suitable for your analysis try another!!!")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------s
                if click=='support vector machine':
                    try:
                        st.write("support vector machine model:")
                        svm=SVC(kernel='rbf')
                        st.write(svm)
                        svm.fit(X_train,y_train)
                        st.success("Model created & fitted successfully!!!")
                        a=svm.score(X_train,y_train)*100
                        y_predlog=svm.predict(X_test)
                        st.write("Y_pred:",y_predlog)
                        st.success("Model has predictted successfully!!!")
                        acrr=accuracy_score(y_test,y_predlog)*100
                        st.write("Accuracy:",a)
                        st.success("Accuracy calculated successfully!!!")
                    except Exception:
                        st.error("This model is not suitable for your analysis try another!!!")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------s
                if click=='K-nearest Neighbor':
                    try:
                        st.write("K-nearest Neighbor Model:")
                        from sklearn.neighbors import KNeighborsClassifier
                        classifier = KNeighborsClassifier(n_neighbors = 8)
                        classifier.fit(X_train, y_train)
                        st.write(classifier)
                        st.success("Model created & fitted successfully!!!")
                        a=classifier.score(X_train,y_train)*100
                        y_predlog=classifier.predict(X_test)
                        st.write("Y_pred:",y_predlog)
                        st.success("Model has predictted successfully!!!")
                        acrr=accuracy_score(y_test,y_predlog)*100
                        st.write("Accuracy:",a)
                        st.success("Accuracy calculated has predictted successfully!!!")
                    except Exception:
                        st.error("This model is not suitable for your analysis try another!!!")
#------------------------------------------------------------------------------------------------------------------------
                if click=='Random Forest':
                    try:
                        st.write("Random Forest Model:")
                        RF=RandomForestClassifier(n_estimators=300)
                        st.write(RF)
                        st.success("Model created & fitted successfully!!!")
                        RF.fit(x_train,y_train)
                        y_pred=RF.predict(x_test)
                        st.dataframe(y_pred)
                        st.success("Model has predicted sucessfully!!!")
                        a=RF.score(X_train,y_train)*100
                        y_predlog=log.predict(X_test)
                        st.write("Y_pred:",y_predlog)
                        st.success("Model has predictted successfully!!!")
                        acrr=accuracy_score(y_test,y_predlog)*100
                        st.write("Accuracy:",a)
                        st.success("Accuracy calculated has predictted successfully!!!")
                    except Exception:
                        st.error("This model is not suitable for your analysis try another!!!")
#======================================================================================================================================================================
            click=st.checkbox('Unsupervised:')
            if click==True:
                st.write("updating soon")
##                b=('Kmeans','PCA')
##                click= st.selectbox("Select Model Type:",b)
###--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##                if click=='Kmeans':
##                    st.write("Kmeans Clustering Model:")
##                    st.write("updating soon")
##                    from sklearn.cluster import KMeans
##                    name = st.text_input("Enter number of cluster yo want:")
##                    kmeans = KMeans(n_clusters=name)
##                    kmeans.fit(training_feature)
##                    st.write(kmeans)
##                    kmeans.fit(X_train)
##                    st.success("Model created & fitted successfully!!!")
##                    st.dataframe(kmeans.labels_)
##                    st.success("Model has predictted successfully!!!")
##                Sir/Ma'am, Yesterday I'd submitted my Day-23 &24 Activity Report for evaluation. Day-24 Activity report got evaluated but Day-23 not got evaluated yet!!! Please note & evaluate it!!! . 
