from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, auc
from sklearn.model_selection import *
import lime
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
import tensorflow as tf
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
    return pd.read_csv('https://raw.githubusercontent.com/tipemat/datasethoracic/main/DateToracic.csv',header = 0)


data = get_data()
st.sidebar.subheader("Prezentare generală a aplicației")
btn_afis_general=st.sidebar.button("Vizualizare prezentare generală")


choose_tabel=st.sidebar.button("Prezentare set de date")
if(choose_tabel):
    st.subheader("Set de date")
    st.write("Setul de date utilizat pentru predicția problemelor toracice este unul furnizat de "
             "către Clinica de Chirurgie Toracică a Spitalului Universitar Municipal din Timișoara, "
             "cuprinzând informații reale de la pacienți.")
    df = get_data()
    st.dataframe(df, height=450, hide_index=True)

    st.write("\n"
             "Studiul este realizat pe 100 de pacienți "
             "(55 de pacienți care nu au probleme toracice și 45 de pacienți care suferă de probleme toracice).")
    fig = plt.figure(figsize=(10, 4))
    sns.countplot(x='ExamHP', data=data, palette= 'hls')
    st.pyplot(fig)

choose_MecanismAtentie = st.sidebar.button("Metoda LIME + Mecanismul de Atentie")
if choose_MecanismAtentie:
    st.header("Modelul dezvoltat")
    st.write("**1) Metoda de explicabilitate LIME (Local Interpretable Model-agnostic Explanations)**")

    with st.expander("Descrierea metodei LIME"):
        st.write("- Este o tehnică utilizată în învățarea automată pentru a explica deciziile luate "
                "de modelele complexe de predicție.")
        st.write("- Scopul principal al metodei LIME este "
                "de a oferi o explicație locală pentru o anumită predicție făcută de un model, "
                "de a identifica care sunt caracteristicile care au contribuit cel mai mult la luarea unei decizii.")

    with st.expander("Descrierea algoritmi utilizati"):
        st.write("- **Random Forest Regressor** este un algoritm de învățare supervizată utilizat pentru regresie. "
                 "Este bazat pe conceptul de **Random Forest**, care este o combinație de mai mulți arbori de decizie, "
                 "fiecare dintre aceștia contribuind la predicția finală.")
        st.write("- **Avantajele** utilizarii acestui algoritm: "
                "~ Poate **furniza o măsură a importanței caracteristicilor** în cadrul setului de date. "
                "Aceasta măsură se bazează pe cât de mult crește eroarea de predicție "
                "atunci când se amestecă valorile unei caracteristici în setul de date."
                "~ Suprapunerea între arborii de decizie din Random Forest ajută la **reducerea varianței** și la **evitarea supraînvățării**.")
    st.write("**Rezultate obtinute in urma aplicarii acestei metode de eplicabilitate**")
    st.write("- Precum se poate observa din diagrama de mai jos, exista caractersitici care aduc un aport important "
             "in ajutarea obtinerii unei predictii mai bune pentru probelemele de toracitate.")
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
    exp = explainer.explain_instance(X_test.iloc[2].values, model.predict)
    strings = ""
    for feature, importance in exp.as_list():
        if importance > 0:
            string_feature = str(feature)
            caracteristica = string_feature.split(" ")
            if ("< " in string_feature):
                positive_features.append(caracteristica[2])
            else:
                positive_features.append(caracteristica[0])

    for i in range(len(positive_features)-1):
        strings = strings + positive_features[i] + ", "
    strings = strings + positive_features[len(positive_features)-1]
    st.write("- Caractersiticile relevante pe care le-am utilizat pentru a imbunatatii modelul sunt: " + strings + ".")
    X_nou = df[positive_features].copy()
    positive_features.append('ExamHP')
    fig = exp.as_pyplot_figure(label=1)
    st.pyplot(fig)

    st.write("**2) Mecanismul de atentie**")

    with st.expander("Descrierea mecanismului"):
        st.write("- Este o tehnică utilizata pentru a permite modelului să se concentreze asupra părților relevante "
                 "ale datelor de intrare în procesul de învățare.")
        st.write("- Scopul principal al mecanismului este "
                 "de a contribui la îmbunătățirea performanței și la înțelegerea mai profundă a datelor de intrare")

    with st.expander("Descrierea algoritmului utilizat"):
        st.write("- Model de predictie dezvoltat cuprinde doua tipuri de layere si anume: **Dense si BatchNormalization**")
        st.write("- Layer-ul **Dense** este un tip de strat în rețelele neurale, care conectează fiecare neuron din stratul curent la toți neuroni "
                 "din stratul următor. Este un strat fully connected, unde fiecare neuron primește toate valorile de intrare de la stratul anterior "
                 "și produce o valoare de ieșire.")
        st.write("- Layer-ul **BatchNormalization** este un strat folosit în rețelele neurale profunde pentru a normaliza activările "
                 "între straturi în timpul procesului de antrenare. A fost introdus pentru a ajuta la accelerarea antrenării, "
                 "la reducerea supraînvățării și la îmbunătățirea generalizării modelului.")
    st.write("- In urma antrenarii modelului s-au obtinut urmatoarele performante: ")

    df_nou = df[positive_features].copy()
    X_train_nou, X_test_nou, y_train_nou, y_test_nou = train_test_split(X_nou, y, test_size=0.2, random_state=0)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=([df_nou.shape[1]-1])),
        tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train_nou, y_train_nou, epochs=10, batch_size=32, validation_data=(X_test_nou, y_test_nou))

    y_pred_nou = model.predict(X_test_nou)
    y_pred_nou = y_pred_nou>0.45

    np.set_printoptions()

    cm = confusion_matrix(y_test_nou, y_pred_nou)
    ac = accuracy_score(y_test_nou, y_pred_nou)
    print(cm)
    print(ac)

    col1, col2 = st.columns(2, gap='large')

    with col1:
        st.write("a) **Matricea de confuzie** cuprinde o reprezentare tabulară a rezultatelor clasificării, "
                 "comparând predicțiile modelului cu valorile reale ale etichetelor claselor.")

        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
        disp.plot()
        plt.title('Matricea de Confuzie')
        plt.show()
        st.pyplot()

        st.write("1) True Positive (TP): Reprezintă numărul de exemple pentru care modelul a prezis corect clasa pozitivă.")
        st.write("2) True Negative (TN): Reprezintă numărul de exemple pentru care modelul a prezis corect clasa negativă.")
        st.write("3) False Positive (FP): Reprezintă numărul de exemple pentru care modelul a prezis greșit clasa pozitivă "
                 "(a prezis că aparține clasei pozitive când, în realitate, nu aparține).")
        st.write("4) False Negative (FN): Reprezintă numărul de exemple pentru care modelul a prezis greșit clasa negativă "
                 "(a prezis că aparține clasei negative când, în realitate, aparține clasei pozitive).")
    with col2:
        st.write("b) **Curba ROC** (Receiver Operating Characteristic) este o metodă utilizată pentru evaluarea performanței "
                 "unui model de clasificare binară. Ea reprezintă o reprezentare grafică a raportului dintre rata de detectare "
                 "a adevăratelor pozitive (True Positive Rate) și rata de detectare a falselor pozitive (False Positive Rate) "
                 "pe măsură ce se modifică pragurile de decizie ale modelului.")
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

        st.write("- O curba ROC ideală se apropie de colțul din stânga sus al graficului, indicând un model cu TPR ridicat și FPR scăzut.")
        st.write("- Cu cât aria sub curba ROC (AUC - Area Under the Curve) este mai mare, cu atât performanța modelului este considerată mai bună.")
        st.write("- Curba ROC oferă o modalitate de a evalua tranzacția între ratele de detectare a adevăratelor pozitive și ratele de detectare a falselor pozitive, permițând alegerea pragului de decizie optim pentru clasificare.")

choose_modelClient = st.sidebar.checkbox("Antreneaza un set de date propriu")
if choose_modelClient:
    st.header("Model incarcat de catre utilizator")
    uploaded_file = st.file_uploader("Choose a file", type = ['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("- Setul de date incarcat de catre dumneavoastra este:")
        st.write(df)
        df = df[df.columns].replace(
            {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'NO': 0, 'YES': 1})
        df = df.select_dtypes(include=['int', 'float'])
        st.write("Noul set de date care va fi utilizat pentru predictie este:")
        st.write(df)

        caracteristica_y = st.selectbox("Alege eticheta corespunzatoare setului de date", df.columns)

        if caracteristica_y:
            X = df.iloc[:, df.columns != caracteristica_y].values
            y = df.iloc[:, df.columns == caracteristica_y].values
            feature = df.drop(columns=caracteristica_y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            option = st.selectbox('Alege clasificatorul utilizat pentru antrenarea modelului',
                     ('Logistic Regression', 'Support Vector Machines', 'Decision Trees', 'Random Forests', 'Naive Bayes'))

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
                                                           class_names=[caracteristica_y], verbose=True, mode='regression',
                                                           discretize_continuous=True)
            positive_features = []
            caractersitica_relevanta = st.slider('Alege caractersitica cea mai relevanta', 0, df.shape[1]-2, 1)
            exp = explainer.explain_instance(X_test[int(caractersitica_relevanta)], model.predict, num_features=df.shape[1]-1)
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
                "- Caractersiticile relevante pe care le-am utilizat pentru a imbunatatii modelul sunt: " + strings + ".")
            X_nou = df[positive_features].copy()
            positive_features.append(caracteristica_y)
            fig = exp.as_pyplot_figure(label=1)
            st.pyplot(fig)

            df_nou = df[positive_features].copy()
            X_train_nou, X_test_nou, y_train_nou, y_test_nou = train_test_split(X_nou, y, test_size=0.2, random_state=0)
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation="relu", input_shape=([df_nou.shape[1] - 1])),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            model.fit(X_train_nou, y_train_nou, epochs=10, batch_size=32, validation_data=(X_test_nou, y_test_nou))

            y_pred_nou = model.predict(X_test_nou)
            y_pred_nou = y_pred_nou > 0.5

            np.set_printoptions()

            cm = confusion_matrix(y_test_nou, y_pred_nou)
            ac = accuracy_score(y_pred_nou, y_test_nou)
            print(cm)
            print(ac)

            st.write("- Rezultate obtinute in urma antrenarii modelului:")
            col1, col2 = st.columns(2, gap='large')

            with col1:
                disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
                disp.plot()
                plt.title('Matricea de Confuzie')
                plt.show()
                st.pyplot()

                st.write(
                    "1) True Positive (TP): Reprezintă numărul de exemple pentru care modelul a prezis corect clasa pozitivă.")
                st.write(
                    "2) True Negative (TN): Reprezintă numărul de exemple pentru care modelul a prezis corect clasa negativă.")
                st.write(
                    "3) False Positive (FP): Reprezintă numărul de exemple pentru care modelul a prezis greșit clasa pozitivă "
                    "(a prezis că aparține clasei pozitive când, în realitate, nu aparține).")
                st.write(
                    "4) False Negative (FN): Reprezintă numărul de exemple pentru care modelul a prezis greșit clasa negativă "
                    "(a prezis că aparține clasei negative când, în realitate, aparține clasei pozitive).")
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
                    "- O curba ROC ideală se apropie de colțul din stânga sus al graficului, indicând un model cu TPR ridicat și FPR scăzut.")
                st.write(
                    "- Cu cât aria sub curba ROC (AUC - Area Under the Curve) este mai mare, cu atât performanța modelului este considerată mai bună.")
                st.write(
                    "- Curba ROC oferă o modalitate de a evalua tranzacția între ratele de detectare a adevăratelor pozitive și ratele de detectare a falselor pozitive, permițând alegerea pragului de decizie optim pentru clasificare.")

if(btn_afis_general or ((not choose_MecanismAtentie) and (not choose_tabel) and (not choose_modelClient))):
        st.title("Model de învățare automată pentru diagnoza toracică îmbunătățit prin metode de explicabilitate")
        st.write("Problemele toracice sunt afecțiuni medicale care afectează zona toracică a corpului, adică zona din spatele sternului și cuprinde inima, plămânii, traheea, bronhiile,"
                 "esofagul și alte structuri ale sistemului respirator și cardiovascular. Aceste probleme "
                 "pot fi cauzate de o varietate de afecțiuni, cum ar fi infecții respiratorii, afecțiuni "
                 "pulmonare, afecțiuni cardiace, boli autoimune sau tulburări de anxietate și pot varia "
                 "între simptome și gravitate. \n"
                 "\n"
                 "Prin această aplicație doresc să prezint un model de învățare automată care este "
                 "antrenat pentru predicția acestor probleme și să cresc nivelul de acuratețe prin "
                 "utilizarea metodelor de explicabilitate. Am ales să utilizez mecanismul de atentie "
                 "pentru a putea obține o pondere mai mare în urma antrenării setului de date. \n"
                 "\n"
                 "Acuratețea modelului antrenat a ajuns la valoarea de peste 0,8%. "
                 "Pentru a putea analiza și explica fiecare caracteristică, am ales să utilizez LIME (Local "
                 "Interpretable Model-Agnostic Explanations), care este o metodă de interpretare a "
                 "modelelor de învățare automată, prezentănd deciziile luate de către un model.")



st.sidebar.write('')
st.sidebar.write('Absolvent: **Costi Flavia-Emanuela**')
st.sidebar.write('Profesor coordonator: **Conf. Dr. Habil. Darian M. Onchiș**')
st.sidebar.write(
        'Universitatea de Vest Timisoara, Facultatea de Matematică și Informatică, Specializarea: BioInformatică' + '\n')
st.sidebar.image('img/Logo-emblema-UVT-14.png', width=55)
st.sidebar.write('2022-2023')
st.set_option('deprecation.showPyplotGlobalUse', False)