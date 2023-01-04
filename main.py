import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

TRAIN = 'train.csv'


def get_forest(X, y):
    parameters = {'n_estimators': [40, 80, 100, 120, 150, 200], 'criterion': ('gini', 'entropy', 'log_loss'),
                  'max_depth': [None, 5, 20, 40]}
    forrest = RandomForestClassifier(verbose=True)
    clf = GridSearchCV(forrest, parameters, verbose=True)
    clf.fit(X, y)
    return clf


def get_svm(X, y):
    parameters = {'kernel': ('poly', 'rbf'), 'C': [1, 10, 50, 100]}
    svc = svm.SVC(verbose=True)
    clf = GridSearchCV(svc, parameters, verbose=True)

    clf.fit(X, y)

    return clf


def get_nn(X, y):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(X.shape[1],)))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss=tf.keras.losses.MeanSquaredError()
                  , optimizer='Adam', metrics=['accuracy'])
    model.summary()
    model.fit(X, y, verbose=1, epochs=100)
    return model


def extract_features(df):
    df['Destination'] = pd.Categorical(df['Destination']).codes
    df['HomePlanet'] = pd.Categorical(df['HomePlanet']).codes
    df = df.fillna(-1)
    features = np.array(df[['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt',
                            'ShoppingMall', 'Spa', 'VRDeck']]).astype(int)

    return features


def main():
    df = pd.read_csv(TRAIN)

    y = np.array(df['Transported']).astype(int)
    X = extract_features(df)

    X_train, X_val, y_train, y_val = train_test_split(X, y)

    # svm = get_svm(X_train, y_train)
    # forest = get_forest(X_train, y_train)
    nn = get_nn(X_train, y_train)

    # preds_svm = svm.predict(X_val)
    # preds_forest = forest.predict(X_val)
    preds_nn = (nn.predict(X_val) > 0.5).astype("int32")
    print(preds_nn, preds_nn.shape)

    # bacc_svm = balanced_accuracy_score(y_val, preds_svm)
    # bacc_forest = balanced_accuracy_score(y_val, preds_forest)
    bacc_nn = balanced_accuracy_score(y_val, preds_nn)

    # print("Bacc SVM: ", bacc_svm)
    # print("Bacc Forest: ", bacc_forest)
    print("Bacc NN: ", bacc_nn)


if __name__ == '__main__':
    main()
