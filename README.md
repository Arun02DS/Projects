# Projects -------> Classification code with pipeline

### "data = pd.read_csv('bank-full.csv', sep=';')
target = data.pop('y')
target = target.map({'yes': 1, 'no':0})
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1234)


categorical_mask = (data.dtypes=='object')
categorical_columns = data.columns[categorical_mask].tolist()
num_cols = data.select_dtypes(include=['int64','float64']).columns.tolist()
oe_cols = [c for c in categorical_columns if data[c].nunique()>5]
ohe_cols = [c for c in categorical_columns if data[c].nunique()<=5]
len(oe_cols), len(ohe_cols), len(num_cols)


ohe_unique_list = [data[c].unique().tolist() for c in ohe_cols]
oe_unique_list = [data[c].unique().tolist() for c in oe_cols]
ohe = OneHotEncoder(categories=ohe_unique_list)
oe = OrdinalEncoder(categories=oe_unique_list)
imp = SimpleImputer(strategy='constant', fill_value=0)


preprocess = make_column_transformer(
    (oe, oe_cols),
    (ohe, ohe_cols),
    (imp, num_cols),
    remainder='passthrough'
)
estimator = XGBClassifier(learning_rate=0.05, max_depth=3, n_estimators=2500, random_state=1234)
fs = SelectKBest(score_func=f_classif, k=5)
selector = RFE(estimator, n_features_to_select=5, step=1)
steps = [
    ('preprocess', preprocess),
    ('select', fs),
    ('clf', estimator)
]
pipeline = Pipeline(steps)


pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
pred_df = pd.DataFrame({'y': y_test,'y_pred': y_pred})
gini = 2*roc_auc_score(y_test, y_pred)-1


param_grid = {
    'clf__learning_rate': np.arange(0.05, 1, 0.05),
    'clf__max_depth': np.arange(3,10,1),
    'clf__n_estimators': np.arange(50,250,50)
}
rand_auc = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=5, scoring='roc_auc', cv=5, verbose=False)
rand_auc.fit(X_train, y_train)
rand_auc.best_score_
y_pred = rand_auc.predict(X_test)
pred_df = pd.DataFrame({'y': y_test,'y_pred': y_pred})
gini = 2*roc_auc_score(y_test, y_pred)-1


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()
    ]
for classifier in classifiers:
    steps = [
        ('preprocess', preprocess),
        ('select', fs),
        ('clf', classifier)
    ]
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)   
    print(classifier)
    print("model score: %.3f" % pipeline.score(X_test, y_test)) " ###



