
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV


class ModelTrainer:

    def TrainModel(self, x, y, preprocessor):

        # train test split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        # full pipeline (preprocessing + model)
        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model",LogisticRegression(class_weight='balanced',random_state=42))
            ]
        )

        # train model
        model_pipeline.fit(x_train, y_train)

        # prediction
        y_pred = model_pipeline.predict(x_test)

        # accuracy
        precision= precision_score(y_test, y_pred)
        recall=recall_score(y_test,y_pred)

        print("Model precision:", precision)
        print("Model recall:",recall)
        fold=KFold(n_splits=5,shuffle=True,random_state=42)
        cross=cross_val_score(model_pipeline,x,y,cv=fold,scoring="roc_auc")

        print("cross_validation_score:",cross.mean())

        params = {
            "model__C":[0.001,0.01,0.1,0.2,0.5,1,5,10],
            }
        grid=GridSearchCV(model_pipeline,params,cv=5,scoring='roc_auc')
        grid.fit(x_train,y_train)
        best_model=grid.best_estimator_

        y_pred1=best_model.predict(x_test)
        print("best_grid_parameters",grid.best_params_)
        print("best_grid_score",grid.best_score_)
        
        y_prob=best_model.predict_proba(x_test)[:,1]
        y_pred2=(y_prob>=0.4).astype('int')

        print("Improved Precision :",precision_score(y_test,y_pred2))
        print("Improved Recall :",recall_score(y_test,y_pred2))
        print("ROC-AUC Score:",roc_auc_score(y_test,y_prob))
        print("Confusion Metric:",confusion_matrix(y_test,y_pred2))

    
        # create models folder
        os.makedirs("models", exist_ok=True)

        # save model
        joblib.dump(best_model, "models/model.pkl")

        print("Model saved successfully!")

        return best_model
