import numpy as np

from imports import *
from assignments.assignment2.b_regression import *
from assignments.assignment2.a_classification import *


"""
The below method should:
?? subtask1  Handle any dataset (if you think worthwhile, you should do some pre-processing)
?? subtask2  Generate a (classification, regression or clustering) model based on the label_column 
             and return the one with best score/accuracy

The label_column can be categorical, numerical or None
-If categorical, run through ML classifiers in "a_classification" file and return the one with highest accuracy: 
    DecisionTree, RandomForestClassifier, KNeighborsClassifier or NaiveBayes
-If numerical, run through these ML regressors in "b_regression" file and return the one with least MSE error: 
    svm_regressor_1(), random_forest_regressor_1()
-If None, run through simple_k_means() and custom_clustering() and return the one with highest silhouette score.
(https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
"""


def generate_model(df: pd.DataFrame, label_column: Optional[pd.Series] = None) -> Dict:
    # model_name is the type of task that you are performing.
    # Use sensible names for model_name so that we can understand which ML models if executed for given df.
    # ex: Classification_random_forest or Regression_SVM.
    # model is trained model from ML process
    # final_score will be Accuracy in case of Classification, MSE in case of Regression and silhouette score in case of clustering.
    # your code here.

    if label_column.any():
        if label_column.dtype == np.number:
            # If the unique values are more than 70% of the total dataset, it will be a regression problem
            if len(label_column.unique()) > (len(label_column)*70 / 100):
                for nc_column in get_numeric_columns(df):
                    df.loc[:, nc_column] = normalize_column(df.loc[:, nc_column])
                #     Please remove drop year from file b_regression as it is data specific
                result1, result2 = run_regression_models_1(df, label_column)
                if(result1["mse"] <= result2["mse"]):
                    return dict(model_name=result1["model"], model=result1["model"], final_score=result1["mse"])
                else:
                    return dict(model_name=result2["model"], model=result2["model"],
                                final_score=result2["mse"])
            else:
                for cc in get_text_categorical_columns(df):
                    le = generate_label_encoder(df.loc[:, cc])
                    df = replace_with_label_encoder(df, cc, le)

                # le = generate_label_encoder(label_column)
                # df = replace_with_label_encoder(df, label_column, le)
                result1, result2, result3, result4 = run_classification_models(df, label_column)
                print("result1: ", result1)
                print("result2: ", result2)
                print("result3: ", result3)
                print("result4: ", result4)


                new_dict = {result1["model"]:result1["accuracy"],result2["model"]:result2["accuracy"],result3["model"]:result3["accuracy"],result4["model"]:result4["accuracy"]}

                max_key = max(new_dict, key=new_dict.get)
                max_value = max(new_dict.values())

                return dict(model_name=max_key, model=max_key, final_score=max_value)

        else:
            for cc in get_text_categorical_columns(df):
                le = generate_label_encoder(df.loc[:, cc])
                df = replace_with_label_encoder(df, cc, le)

            result1, result2, result3, result4 = run_classification_models(df, label_column)
            print("result1: ", result1)
            print("result2: ", result2)
            print("result3: ", result3)
            print("result4: ", result4)

            new_dict = {result1["model"]: result1["accuracy"], result2["model"]: result2["accuracy"],
                        result3["model"]: result3["accuracy"], result4["model"]: result4["accuracy"]}

            max_key = max(new_dict, key=new_dict.get)
            max_value = max(new_dict.values())

            return dict(model_name=max_key, model=max_key, final_score=max_value)

    return dict(model_name=None, model=None, final_score=None)


def run_custom():
    start = time.time()
    print("Custom modeling in progress...")

    # Assuming dataset will only contain train and label will contain the column, as it is mention in the function definition, pd.Series will be passed

    # Categorical Data
    df = pd.DataFrame(read_dataset(Path('..', '..', 'iris.csv')))  # Markers will run your code with a separate dataset unknown to you.
    print(df)
    result = generate_model(df,df["species"])


    # # Numeric data
    # df = pd.DataFrame(read_dataset(Path('..', '..', 'pp_gas_emission', 'gt_2011.csv')))
    # result = generate_model(df, df["NOX"])

    print(f"result:\n{result}\n")

    end = time.time()
    run_time = round(end - start)
    print("Custom modeling ended...")
    print(f"{30 * '-'}\nCustom run_time:{run_time}s\n{30 * '-'}\n")


if __name__ == "__main__":
    run_custom()
