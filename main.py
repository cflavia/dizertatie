from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from lime import lime_tabular

import lime
import numpy as np
import pandas as pd
import qrcode
import seaborn as sns
import streamlit as st
import tensorflow as tf

apptitle = 'Application'
st.set_page_config(page_title=apptitle, page_icon=":bar_chart:")

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-color: #F63366 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def get_data():
    return pd.read_csv('https://raw.githubusercontent.com/tipemat/datasethoracic/main/DateToracic.csv', header=0)


def get_data_filter():
    return pd.read_csv('https://raw.githubusercontent.com/tipemat/datasethoracic/main/DateToracic.csv', header=0)


def codQR():
    link = "https://cflavia-dizertatie-main-nhrwqh.streamlit.app/"  # Link-ul către aplicație

    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=10, border=4)
    qr.add_data(link)
    qr.make(fit=True)

    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_img.save("qrcode.png")


def load_data_map():
    data = pd.read_csv('resources/covid.csv')
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    return data


def get_data_predict():
    return pd.read_csv('resources/diabetes_data_upload.csv')


data = get_data()
st.sidebar.subheader("Overview of the application")
btn_afis_general = st.sidebar.button("View overview")

choose_tabel = st.sidebar.button("Overview dataset")
if (choose_tabel):
    st.subheader("Dataset")
    st.write(
        "The dataset used for predicting chest problems is one provided by the Clinic of Thoracic Surgery at the "
        "Municipal University Hospital in Timișoara, containing real information from 100 patients.")
    df = get_data()
    df = df.drop(columns="No")
    st.dataframe(df, height=450, hide_index=True)

    st.write("\n"
             "The study is conducted on 100 patients (55 patients without chest problems and 45 patients who have "
             "been diagnosed with chest problems).")
    fig = plt.figure(figsize=(10, 4))
    sns.countplot(x='ExamHP', data=df, palette='hls')
    st.pyplot(fig)

    df.rename(columns={'Sem0l1 ': 'Sem0l1'}, inplace=True)

    for col in df.columns:
        df.loc[(df["ExamHP"] == 0) & (df[col].isnull()), col] = df[df["ExamHP"] == 0][col].median()
        df.loc[(df["ExamHP"] == 1) & (df[col].isnull()), col] = df[df["ExamHP"] == 1][col].median()
    st.write(
        "<div style='text-align:justify;font-size: 16px;'>Below, you can view the histogram with values for each of the components that influence the thoracic diagnosis"
        "<li>The more spread out the points are, the more diverse the values are. Places where they are closely connected indicate that the respective values are close, representing a majority.</li>"
        "<li style='color: orange'>ExamHP: 0.0 - Individuals who do not have thoracic issues</li>"
        "<li style='color: red'>ExamHP: 1.0 - Individuals who have thoracic issues</li></div>",
        unsafe_allow_html=True)
    fig, axes = plt.subplots(9, 1, figsize=(20, 5))

    for col in df.columns:
        if col != "ExamHP":
            st.write('\n')
            sns.catplot(x="ExamHP", y=col, data=df, hue='ExamHP', palette=sns.color_palette(['orange', 'red']),
                        height=5.5, aspect=11.7 / 6.5, legend=True)
            st.write("Scatter plot for " + col + "in both cases: whether they have thoracic problems or not.")
            st.pyplot()

    scaler = MinMaxScaler()
    fig = plt.figure(figsize=(20, 15))
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")

    plt.title("Correlation Matrix")
    plt.xlabel("Variable")
    plt.ylabel("Variable")
    plt.show()
    st.pyplot(fig)

choose_MecanismAtentie = st.sidebar.button("LIME Method + Attention Mechanism")
if choose_MecanismAtentie:
    st.header("Model Structure")
    st.write("**1) LIME (Local Interpretable Model-agnostic Explanations) explainability method**")

    with st.expander("The description of the LIME method:"):
        st.write(
            "- It is a technique used in machine learning to explain the decisions made by complex prediction models.")
        st.write("- The primary objective of the LIME method is to provide a localized explanation for a particular "
                 "prediction generated by a model, elucidating the features that exerted the most significant "
                 "influence on the decision.")

    with st.expander("Algorithm description"):
        st.write("- **Random Forest Regressor** It is a supervised learning algorithm used for regression."
                 " It is based on the concept of Random Forest, which is a combination of multiple decision trees,"
                 " each of them contributing to the final prediction.")
        st.write("- **The advantages ** of using this algorithm:"
                 "~ It can **provide a measure of feature importance** within the dataset. "
                 "This measure is based on how much the prediction error increases when the values of "
                 "a feature are shuffled within the dataset"
                 "~ The overlap between decision trees in a Random Forest helps **reduce variance ** "
                 "and **avoid overfitting**.")
    st.write("**The results obtained after applying this explainability method**")
    st.write("- As can be observed from the diagram below, there are features that make a significant contribution to "
             "achieving a better prediction for chest problems.")
    df = get_data()
    df = df.drop(columns="No")
    df.rename(columns={'Sem0l1 ': 'Sem0l1'}, inplace=True)

    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    X = df.drop('ExamHP', axis=1)
    y = df['ExamHP']
    feature = df.drop(columns='ExamHP')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, mode="regression", feature_names=X_train.columns)

    positive_features = []
    exp = explainer.explain_instance(X_test.iloc[2].values, model.predict, num_features=len(X_test))
    st.pyplot(exp.as_pyplot_figure(label=1))
    strings = ""
    for feature, importance in exp.as_list():
        if importance > 0:
            string_feature = str(feature)
            caracteristica = string_feature.split(" ")
            if ("< " in string_feature):
                positive_features.append(caracteristica[2])
            else:
                positive_features.append(caracteristica[0])

    for i in range(len(positive_features) - 1):
        strings = strings + positive_features[i] + ", "
    strings = strings + positive_features[len(positive_features) - 1]
    st.write("- The relevant features we have used to improve the model are:" + strings + ".")
    X_nou = df[positive_features].copy()
    positive_features.append('ExamHP')

    st.write("**2) The attention mechanism**")

    with st.expander("The description of the mechanism"):
        st.write("- It is a technique used to enable the model to focus on the relevant parts of the input "
                 "data during the learning process")
        st.write("- The primary purpose of the mechanism is to contribute to improving performance and gaining a deeper"
                 " understanding of the input data.")

    with st.expander("The description of the algorithm used."):
        st.write(
            "- The developed prediction model consists of two types of layers, namely: **Dense și BatchNormalization**")
        st.write(
            "- The **Dense** layer is used in neural networks, where each neuron in the current layer connects to all "
            "neurons in the next layer. It is a fully connected layer, where each neuron receives all input values from "
            "the previous layer and produces an output value.")
        st.write(
            "- The **BatchNormalization** layer is a layer used in deep neural networks to normalize activations"
            " between layers during the training process. It was introduced to help speed up training, "
            "reduce overfitting, and improve the model's generalization.")
    st.write("- After training the model, the following results were achieved:")

    df_nou = df[positive_features].copy()
    X_train_nou, X_test_nou, y_train_nou, y_test_nou = train_test_split(X_nou, y, test_size=0.2, random_state=0)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=([df_nou.shape[1] - 1])),
        tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train_nou, y_train_nou, epochs=10, batch_size=32, validation_data=(X_test_nou, y_test_nou))

    y_pred_nou = model.predict(X_test_nou)
    y_pred_nou = y_pred_nou > 0.45

    np.set_printoptions()

    cm = confusion_matrix(y_test_nou, y_pred_nou)
    ac = accuracy_score(y_test_nou, y_pred_nou)

    col1, col2 = st.columns(2, gap='large')

    with col1:
        st.write("a) The **confusion matrix** provides a tabular representation of classification results, "
                 "comparing the model's predictions with the actual values of class labels.")

        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
        disp.plot()
        plt.title('Confusion matrix')
        plt.show()
        st.pyplot()

        st.write(
            "1) True Positive (TP): Represents the number of instances for which the model correctly predicted "
            "the positive class.")
        st.write(
            "2) True Negative (TN): Represents the number of instances for which the model correctly predicted"
            " the negative class")
        st.write(
            "3) False Positive (FP): Represents the number of instances for which the model incorrectly predicted "
            "the positive class (predicted it belongs to the positive class when, in reality, it doesn't)")
        st.write(
            "4) False Negative (FN): Represents the number of instances for which the model incorrectly \
            predicted the negative class (predicted it belongs to the negative class when, in reality,"
            " it belongs to the positive class).")
    with col2:
        st.write(
            "b) The ROC curve (Receiver Operating Characteristic) is a method used to evaluate the performance of a "
            "binary classification model. It provides a graphical representation of the trade-off between the"
            " true positive rate and the false positive rate as the model's decision thresholds are varied.")
        fpr, tpr, thresholds = metrics.roc_curve(y_test_nou, y_pred_nou)
        plt.axis('scaled')
        plt.xlim([0.1, 0.9])
        plt.ylim([0.1, 0.9])
        plt.title('Curba ROC')
        plt.plot(fpr, tpr, 'b')
        plt.fill_between(fpr, tpr, facecolor='lightblue', alpha=0.5)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        st.pyplot()

        st.write(
            "An ideal ROC curve approaches the upper-left corner of the graph, indicating a model with a high "
            "true positive rate (TPR) and a low false positive rate (FPR).")
        st.write(
            "Greater AUC (Area Under the Curve) indicates better model performance.")
        st.write("- The ROC curve provides a way to assess the trade-off between true positive detection rates and "
                 "false positive detection rates, enabling the selection of the optimal decision"
                 " threshold for classification.")

choose_urologyDataset = st.sidebar.button("Train the urology dataset")
if (choose_urologyDataset):
    data = pd.read_csv('https://raw.githubusercontent.com/cflavia/cancerprostata/main/baza_de_date1.csv', header=0)
    col = ['GLEASON', 'SWE']
    DF = data[col]
    st.title("The multiclass example for urology dataset")
    st.write("The relevant columns for this scenario are: ")
    st.write(DF)

    import random

    columns = ['age', 'PSA', 'VOL.P.', 'EX.HP', 'GLEASON', 'FRAGMENTE PBP', 'FRAGMENTE ADK+', 'INVAZIE PERINEURALA',
               'SWE']
    fragment1 = fragment2 = fragment3 = fragment4 = fragment5 = fragment6 = fragment7 = fragment8 = fragment9 = fragment10 = fragment11 = fragment12 = []

    df = pd.DataFrame(data)
    list1 = [22]

    for i, j in data.iterrows():
        coloana = data.at[i, 'SWE']
        import re

        string_new = re.sub("\s\s+", " ", str(coloana))
        x = string_new.split(" ")
        if (len(x) > 0 and x[0] != 'nan' and x[0] != '' and len(x[0]) < 4):
            fragment1.append(int(x[0]))
        else:
            fragment1.append(random.choice(list1))
    for i, j in data.iterrows():
        coloana = data.at[i, 'SWE']
        import re

        string_new = re.sub("\s\s+", " ", str(coloana))
        x = string_new.split(" ")
        if (len(x) > 1 and x[1] != 'nan' and x[1] != '' and len(x[1]) < 4):
            fragment2.append(int(x[1]))
        else:
            fragment2.append(random.choice(list1))
    for i, j in data.iterrows():
        coloana = data.at[i, 'SWE']
        import re

        string_new = re.sub("\s\s+", " ", str(coloana))
        x = string_new.split(" ")
        if (len(x) > 2 and x[2] != 'nan' and x[2] != ''):
            fragment3.append(int(x[2]))
        else:
            fragment3.append(random.choice(list1))
    for i, j in data.iterrows():
        coloana = data.at[i, 'SWE']
        import re

        string_new = re.sub("\s\s+", " ", str(coloana))
        x = string_new.split(" ")
        if (len(x) > 3 and x[3] != 'nan' and x[3] != ''):
            fragment4.append(int(x[3]))
        else:
            fragment4.append(random.choice(list1))
    for i, j in data.iterrows():
        coloana = data.at[i, 'SWE']
        import re

        string_new = re.sub("\s\s+", " ", str(coloana))
        x = string_new.split(" ")
        if (len(x) > 4 and x[4] != 'nan' and x[4] != ''):
            fragment5.append(int(x[4]))
        else:
            fragment5.append(random.choice(list1))
    for i, j in data.iterrows():
        coloana = data.at[i, 'SWE']
        import re

        string_new = re.sub("\s\s+", " ", str(coloana))
        x = string_new.split(" ")
        if (len(x) > 5 and x[5] != 'nan' and x[5] != ''):
            fragment6.append(int(x[5]))
        else:
            fragment6.append(random.choice(list1))
    for i, j in data.iterrows():
        coloana = data.at[i, 'SWE']
        import re

        string_new = re.sub("\s\s+", " ", str(coloana))
        x = string_new.split(" ")
        if (len(x) > 6 and x[6] != 'nan' and x[6] != '' and len(x[6]) < 4):
            fragment7.append(int(x[6]))
        else:
            fragment7.append(random.choice(list1))
    for i, j in data.iterrows():
        coloana = data.at[i, 'SWE']
        import re

        string_new = re.sub("\s\s+", " ", str(coloana))
        x = string_new.split(" ")
        if (len(x) > 7 and x[7] != 'nan' and x[7] != '' and len(x[7]) < 4):
            fragment8.append(int(x[7]))
        else:
            fragment8.append(random.choice(list1))
    for i, j in data.iterrows():
        coloana = data.at[i, 'SWE']
        import re

        string_new = re.sub("\s\s+", " ", str(coloana))
        x = string_new.split(" ")
        if (len(x) > 8 and x[8] != 'nan' and x[8] != '' and len(x[8]) < 4):
            fragment9.append(int(x[8]))
        else:
            fragment9.append(random.choice(list1))
    for i, j in data.iterrows():
        coloana = data.at[i, 'SWE']
        import re

        string_new = re.sub("\s\s+", " ", str(coloana))
        x = string_new.split(" ")
        if (len(x) > 9 and x[9] != 'nan' and x[9] != '' and len(x[9]) < 4):
            fragment10.append(int(x[9]))
        else:
            fragment10.append(random.choice(list1))
    for i, j in data.iterrows():
        coloana = data.at[i, 'SWE']
        import re

        string_new = re.sub("\s\s+", " ", str(coloana))
        x = string_new.split(" ")
        if (len(x) > 10 and x[10] != 'nan' and x[10] != '' and len(x[10]) < 4):
            fragment11.append(int(x[10]))
        else:
            fragment11.append(random.choice(list1))
    for i, j in data.iterrows():
        coloana = data.at[i, 'SWE']
        import re

        string_new = re.sub("\s\s+", " ", str(coloana))
        x = string_new.split(" ")
        if (len(x) > 11 and x[11] != 'nan' and x[11] != '' and len(x[11]) < 4):
            fragment12.append(int(x[11]))
        else:
            fragment12.append(random.choice(list1))

    df['Fragment1'] = fragment1[:244]
    df['Fragment2'] = fragment2[244:(244 * 2)]
    df['Fragment3'] = fragment3[(244 * 2):(244 * 3)]
    df['Fragment4'] = fragment4[(244 * 3):(244 * 4)]
    df['Fragment5'] = fragment5[(244 * 4):(244 * 5)]
    df['Fragment6'] = fragment6[(244 * 5):(244 * 6)]
    df['Fragment7'] = fragment7[(244 * 6):(244 * 7)]
    df['Fragment8'] = fragment8[(244 * 7):(244 * 8)]
    df['Fragment9'] = fragment9[(244 * 8):(244 * 9)]
    df['Fragment10'] = fragment10[(244 * 9):(244 * 10)]
    df['Fragment11'] = fragment11[(244 * 10):(244 * 11)]
    df['Fragment12'] = fragment12[(244 * 11):]

    df = df.drop(['SWE'], axis=1)
    df[df.columns[8:21]].describe()
    st.write("The dataset after to split SWE column into 12 fragments is:")
    st.write(df)

    cols_to_norm = ['Fragment1', 'Fragment2', 'Fragment3', 'Fragment4', 'Fragment5', 'Fragment6', 'Fragment7',
                    'Fragment8', 'Fragment9', 'Fragment10', 'Fragment11', 'Fragment12']

    st.write("The correlation matrix for urology dataset: ")
    corr = df[cols_to_norm].corr()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style='white')
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr, annot=True, cmap='RdYlGn', vmin=-1, vmax=1, ax=ax)
    ax.set_xlabel('Fragments')
    ax.set_ylabel('Fragments')
    plt.show()
    st.pyplot()

    data_new = {}
    for i in range(0, 20):
        data_new[df.columns[i]] = []
    data_new['Risc'] = []
    df_new = pd.DataFrame(data_new)

    for i, j in df.iterrows():
        coloana_Gleason = df.at[i, 'GLEASON']
        import re

        string_new = re.sub("\s\s+", "=", str(coloana_Gleason))
        valAndRez = string_new.split("=")
        valRez_new = re.sub("\s\s+", "+", str(valAndRez[0]))
        val = valRez_new.split("+")
        if (len(val) < 2):
            break

        if (val[0] == '3' and val[1] == '3'):
            index = len(df_new.index)
            df_new.loc[index] = j
            df_new.at[index, 'ISUP_Score'] = int(1)
            df_new.at[index, 'Risc'] = 0
        if (val[0] == '3' and val[1] == '4'):
            index = len(df_new.index)
            df_new.loc[index] = j
            df_new.at[index, 'ISUP_Score'] = int(2)
            df_new.at[index, 'Risc'] = 0
        if (val[0] == '4' and val[1] == '3'):
            index = len(df_new.index)
            df_new.loc[index] = j
            df_new.at[index, 'ISUP_Score'] = int(3)
            df_new.at[index, 'Risc'] = 1
        if (val[0] == '4' and val[1] == '4'):
            index = len(df_new.index)
            df_new.loc[index] = j
            df_new.at[index, 'ISUP_Score'] = int(4)
            df_new.at[index, 'Risc'] = 1
        if (val[0] == '4' or val[0] == '5') and (val[1] == '5' or val[1] == '4'):
            index = len(df_new.index)
            df_new.loc[index] = j
            df_new.at[index, 'ISUP_Score'] = int(5)
            df_new.at[index, 'Risc'] = 1

    df_new['Risc'], df_new['ISUP_Score'] = df_new['ISUP_Score'], df_new['Risc']
    df_new.rename(columns={'Risc': 'ISUP_Score', 'ISUP_Score': 'Risc'}, inplace=True)
    df_new = pd.concat([df_new] * 4, ignore_index=True)

    st.write("The dataset after add the Risc and ISUP_Score columns:")
    st.write(df_new)

    X = df_new[df_new.columns[9:21]]
    y = df_new['ISUP_Score']

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.80, test_size=0.2, stratify=y,
                                                        random_state=0)
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier()
    model.fit(X_train, Y_train.astype('int'))
    from lime import lime_tabular
    import lime

    y_pred = model.predict(X_test)
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, mode="regression", feature_names=X_train.columns)

    positive_features = []
    exp = explainer.explain_instance(X_test.iloc[1].values, model.predict, num_features=len(X_test))
    exp.as_pyplot_figure(label=1)
    st.pyplot()
    strings = ""
    for feature, importance in exp.as_list():
        if importance > 0:
            string_feature = str(feature)
            caracteristica = string_feature.split(" ")
            if ("< " in string_feature):
                positive_features.append(caracteristica[2])
            else:
                positive_features.append(caracteristica[0])

    for i in range(len(positive_features) - 1):
        strings = strings + positive_features[i] + ", "
    strings = strings + positive_features[len(positive_features) - 1]
    st.write("The relevant features for urology dataset are: " + strings)
    X_nou = df_new[positive_features].copy()
    positive_features.append('ISUP_Score')

    from sklearn.preprocessing import OneHotEncoder

    df_nou = df_new[positive_features].copy()
    y = df_nou['ISUP_Score']
    encoder = OneHotEncoder()

    encoded_Y = encoder.fit(y.values.reshape(-1, 1))
    encoded_Y = encoded_Y.transform(y.values.reshape(-1, 1)).toarray()

    k = 20

    kf = KFold(n_splits=k, random_state=1, shuffle=True)
    model = LogisticRegression(solver='liblinear')
    for train_index, test_index in kf.split(X):
        trainX, testX = X.iloc[train_index, :], X.iloc[test_index, :]
        trainY, testY = encoded_Y[train_index], encoded_Y[test_index]

    sc = StandardScaler()
    trainX = sc.fit_transform(trainX)
    testX = sc.transform(testX)
    valX = trainX
    valY = trainY
    re_transformed_array_trainY = encoder.inverse_transform(trainY)
    unique_elements, counts_elements = np.unique(re_transformed_array_trainY, return_counts=True)
    unique_elements_and_counts_trainY = pd.DataFrame(np.asarray((unique_elements, counts_elements)).T)
    unique_elements_and_counts_trainY.columns = ['ISUP_Score', 'count']

    import tensorflow as tf

    input_shape = trainX.shape[1]

    n_batch_size = 10

    n_steps_per_epoch = int(trainX.shape[0] / n_batch_size)
    n_validation_steps = int(valX.shape[0] / n_batch_size)
    n_test_steps = int(testX.shape[0] / n_batch_size)

    from keras import models
    from keras import layers

    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)), )
    model.add(layers.Dense(256, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(128, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    history = model.fit(trainX,
                        trainY,
                        steps_per_epoch=n_steps_per_epoch,
                        epochs=100,
                        batch_size=n_batch_size,
                        validation_data=(valX, valY),
                        validation_steps=n_validation_steps)
    hist_df = pd.DataFrame(history.history)
    hist_df['epoch'] = hist_df.index + 1
    cols = list(hist_df.columns)
    cols = [cols[-1]] + cols[:-1]
    hist_df = hist_df[cols]
    values_of_best_model = hist_df[hist_df.loss == hist_df.loss.min()]


    def predict_prob(number):
        return [number[0], 1 - number[0]]


    y_prob = np.array(list(map(predict_prob, model.predict(testX))))

    cm = confusion_matrix(trainY.argmax(axis=1), valY.argmax(axis=1))
    plt.figure(figsize=(12, 6))
    plt.title("Confusion Matrix")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel("Actual Values")
    plt.xlabel("Predicted Values")
    plt.show()
    st.pyplot()

    clf = OneVsRestClassifier(LinearSVC(random_state=0))
    y_score = clf.fit(trainX, trainY).decision_function(testX)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(testY[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    st.write("The ROC curve for urology dataset:")

    for i in range(5):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.1, 0.9])
        plt.ylim([0.1, 0.9])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for Risc: {}'.format(i + 1))
        plt.legend(loc="lower right")
        plt.show()
        st.pyplot()

choose_modelClient = st.sidebar.checkbox(
    "Training a custom dataset for multiclass classification.")
if choose_modelClient:
    st.header("The model uploaded by the user.")
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("- The dataset uploaded by you is:")
        st.write(df)
        df = df[df.columns].replace(
            {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'NO': 0, 'YES': 1})
        df = df.select_dtypes(include=['int', 'float'])
        st.write("The new dataset that will be used for making predictions is the following:")
        st.write(df)

        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        caracteristica_y = st.selectbox("Select the dependent variable for the dataset.", df.columns)

        if caracteristica_y:
            X = df.iloc[:, df.columns != caracteristica_y].values
            y = df.iloc[:, df.columns == caracteristica_y].values
            feature = df.drop(columns=caracteristica_y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            option = st.selectbox("Choose the classifier used in model training.",
                                  ('Logistic Regression', 'Support Vector Machines', 'Decision Trees', 'Random Forests',
                                   'Naive Bayes'))

            model = LogisticRegression()
            if option == 'Support Vector Machines':
                model = SVC()
            if option == 'Decision Trees':
                model = DecisionTreeClassifier()
            if option == 'Random Forests':
                model = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=5)
            if option == 'Naive Bayes':
                model = GaussianNB()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature,
                                                               class_names=[caracteristica_y], verbose=True,
                                                               mode='regression',
                                                               discretize_continuous=True)
            positive_features = []
            caractersitica_relevanta = st.slider(
                "Select the most relevant independent feature for you.", 0, df.shape[1] - 2, 1)
            exp = explainer.explain_instance(X_test[int(caractersitica_relevanta)], model.predict,
                                             num_features=df.shape[1] - 1)
            strings = ""
            for feature, importance in exp.as_list():
                if importance > 0:
                    string_feature = str(feature)
                    caracteristica = string_feature.split(" ")
                    if ("< " in string_feature):
                        positive_features.append(caracteristica[2])
                    else:
                        positive_features.append(caracteristica[0])

            for i in range(len(positive_features) - 1):
                strings = strings + positive_features[i] + ", "
            strings = strings + positive_features[len(positive_features) - 1]
            st.write(
                "The relevant features used to improve the model are:" + strings + ".")
            X_nou = df[positive_features].copy()
            positive_features.append(caracteristica_y)
            fig = exp.as_pyplot_figure(label=1)
            st.pyplot(fig)

            df_nou = df[positive_features].copy()
            X_train_nou, X_test_nou, y_train_nou, y_test_nou = train_test_split(X_nou, y, test_size=0.2, random_state=0)
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation="relu", input_shape=([df_nou.shape[1] - 1])),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            model.fit(X_train_nou, y_train_nou, epochs=10, batch_size=32, validation_data=(X_test_nou, y_test_nou))

            y_pred_nou = model.predict(X_test_nou)
            y_pred_nou = y_pred_nou > 0.45

            np.set_printoptions()

            cm = confusion_matrix(y_test_nou, y_pred_nou)
            ac = accuracy_score(y_pred_nou, y_test_nou)

            st.write("- The results obtained after training the model are:")
            col1, col2 = st.columns(2, gap='large')

            with col1:
                disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
                disp.plot()
                plt.title('Confusion matrix')
                plt.show()
                st.pyplot()

                st.write(
                    "1) True Positive (TP): Represents the number of instances for which the model correctly predicted "
                    "the positive class.")
                st.write(
                    "2) True Negative (TN): Represents the number of instances for which the model correctly predicted"
                    " the negative class")
                st.write(
                    "3) False Positive (FP): Represents the number of instances for which the model "
                    "incorrectly predicted "
                    "the positive class (predicted it belongs to the positive class when, in reality, it doesn't)")
                st.write(
                    "4) False Negative (FN): Represents the number of instances for which the model incorrectly \
                    predicted the negative class (predicted it belongs to the negative class when, in reality,"
                    " it belongs to the positive class).")
            with col2:
                fpr, tpr, thresholds = metrics.roc_curve(y_test_nou, y_pred_nou)
                plt.axis('scaled')
                plt.xlim([0.1, 0.9])
                plt.ylim([0.1, 0.9])
                plt.title('Curba ROC')
                plt.plot(fpr, tpr, 'b')
                plt.fill_between(fpr, tpr, facecolor='lightblue', alpha=0.5)
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.show()
                st.pyplot()

                st.write(
                    "An ideal ROC curve approaches the upper-left corner of the graph, indicating a model with a high "
                    "true positive rate (TPR) and a low false positive rate (FPR).")
                st.write(
                    "Greater AUC (Area Under the Curve) indicates better model performance.")
                st.write(
                    "- The ROC curve provides a way to assess the trade-off between true positive detection rates and "
                    "false positive detection rates, enabling the selection of the optimal decision"
                    " threshold for classification.")

if (btn_afis_general or ((not choose_MecanismAtentie) and (not choose_tabel) and (not choose_modelClient) and (
not choose_urologyDataset))):
    st.title("Machine learning model for improving the management of cancer risk groups through explainability methods.")
    st.write(

        "Background: Lung cancers are the most common worldwide and prostate cancers are among the second as frequency, diagnosed in men. The automatic ranking in the risk groups of such diseases is highly in demand, but the clinical practice showed us that for a sensitive screening of the clinical parameters with an artificial intelligence system, a customary defined deep neural network classifier is not sufficient given the usually small size medical datasets."
        "\n\n"
        "Methods: In this paper, we propose a new management of method of cancer risk groups management based on a supervised neural network model that is further enhanced by using an features-attention mecha-nism, in order to boost its level of accuracy. For the analysis of each clinical parameter, we used Local Interpretable Model-Agnostic Explanations, which is a post-hoc model agnostic technique that outlines features importance. After that, we applied the attention mechanism in order to obtain a higher weight after training. We have tested the method on two datasets, one for binary-class in case of thoracic cancer and one for the multi-class classification in case of urology cancers to show the wide availability and versatility of the method."
        "\n\n"
        "Results: The accuracy of the models trained in this way, reached the value of more than 80% for both clinical tasks."
        "\n"
        "\n"
        "Conclusions: Our ex-periments demonstrate that by using explainability results as feedback signals in conjunction with the attention mechanism, we were able to increase the accuracy of the base model with more than 20% on small medical datasets, reaching a critical threshold for providing recommendations based on the collected clinical parameters."
    )

st.sidebar.write('')
st.sidebar.write('Developer: **Flavia-Emanuela Costi**')
st.sidebar.write('Prof.: **Conf. Dr. Habil. Darian M. Onchiș**')
st.sidebar.write(
    "West University of Timisoara - BioInformatics" + '\n')
st.sidebar.image('img/Logo-emblema-UVT-14.png', width=55)
st.sidebar.write('2022-2023')
st.set_option('deprecation.showPyplotGlobalUse', False)
