import os
import sys
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def load_split_scale():
    """
    Loads raw data into pandas dataframe, splits test & training sets, scales & transforms appropriately.
    Returns X_train, X_test, y_train, y_test.
    """


    PROJ_ROOT = os.path.join(os.pardir, os.pardir)

    data_path = os.path.join(PROJ_ROOT,
                        'data',
                        'raw',
                        'diabetes_012_health_indicators_BRFSS2015.csv')

    data1_filename = "diabetes_012_health_indicators_BRFSS2015.csv"
    raw_data_filepath = "../../data/raw/"

    df1 = pandas.read_csv(raw_data_filepath + data1_filename)


    X = df1.drop(columns=['Diabetes_012'])
    y = df1['Diabetes_012']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=137)

    for frame in [X_train, X_test, y_train, y_test]:
        frame.reset_index(drop=True, inplace=True)

    #Select features to be scaled
    nonbinary_columns = [col for col in X_train.columns if df1[col].max() >= 2 and col!="Diabetes_012"]
    binary_columns = [col for col in X_train.columns if df1[col].max() < 2 and col!="Diabetes_012"]

    column_trans = ColumnTransformer([('scaled_columns', StandardScaler(), nonbinary_columns)],
                                    remainder='passthrough')

    new_col_order = nonbinary_columns + binary_columns

    X_train_trans = column_trans.fit_transform(X_train)
    X_train_scaled = pandas.DataFrame(X_train_trans, columns=new_col_order)

    X_test_trans = column_trans.transform(X_test)
    X_test_scaled = pandas.DataFrame(X_test_trans, columns = new_col_order)

    # Rename scaled data for convenience
    X_train = X_train_scaled
    X_test = X_test_scaled

    return X_train, X_test, y_train, y_test


# Column descriptions for quick & easy reference.
# Taken from datacard on Kaggle, augmented by descriptions given in primary source.
column_descriptions = {
    'Diabetes_012':"0 = no diabetes 1 = prediabetes 2 = diabetes",
     'HighBP':"0 = no high BP 1 = high BP",
     'HighChol':"0 = no high cholesterol 1 = high cholesterol",
     'CholCheck':"0 = no cholesterol check in 5 years 1 = yes cholesterol check in 5 years",
     'BMI':"Body Mass Index",
     'Smoker':"Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] 0 = no 1 = yes",
     'Stroke':"(Ever told) you had a stroke. 0 = no 1 = yes",
     'HeartDiseaseorAttack':"coronary heart disease (CHD) or myocardial infarction (MI) 0 = no 1 = yes",
     'PhysActivity':"physical activity in past 30 days - not including job 0 = no 1 = yes",
     'Fruits':"Consume Fruit 1 or more times per day 0 = no 1 = yes ",
     'Veggies':"Consume Vegetables 1 or more times per day 0 = no 1 = yes",
     'HvyAlcoholConsump':"Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) 0 = no",
     'AnyHealthcare':"Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc. 0 = no 1 = yes",
     'NoDocbcCost':"Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? 0 = no 1 = yes",
     'GenHlth':"Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor",
     'MentHlth':"Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? scale 1-30 days",
     'PhysHlth':"Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? scale 1-30 days",
     'DiffWalk':"Do you have serious difficulty walking or climbing stairs? 0 = no 1 = yes",
     'Sex':"0 = female 1 = male",
     'Age':"13-level age category (_AGEG5YR see codebook) 1 = 18-24, 2 = 25-29, 3 = 30-34, etc, 9 = 60-64, 13 = 80 or older",
     'Education':"Education level (EDUCA see codebook) scale 1-6 1 = Never attended school or only kindergarten 2 = Grades 1 through 8 (Elementary) 3 = Grades 9 through 11 (Some high school) 4 = Grade 12 or GED (High school graduate) 5 = College 1 year to 3 years (Some college or technical school) 6 = College 4 years or more (College graduate)",
     'Income':"Income scale (INCOME2 see codebook) scale 1-8. 1 = less than $10k, 2 = less than $15k, 3 = less than $20k, 4 = less than $25k, 5 = less than $35k, 6 = less than $50k, 7 = less than $75k, 8 = $75k or more"
}
