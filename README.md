- Define normalize_data function which takes as arguments the train data, test data and the type of normalization and returns the normalized data

```
scaler.fit(train_data)

scaled_train_data = scaler.transform(train_data)

scaled_test_data = scaler.transform(test_data)

return scaled_train_data, scaled_test_data
```

- Build the vocabulary

```
bow = BagOfWords()

bow.build_vocabulary(training_data)
```

- Turning train_data text in numerical features

```
training_features = bow.get_features(training_data)

testing_features = bow.get_features(test_data)
```

- Normalizing the features to use it in training the model

```norm_train, norm_test = normalize_data(training_features, testing_features, type='l2')```

- Training the SVM model

```
model = svm.SVC(C=1.0, kernel='linear')

model.fit(norm_train, training_labels)
```

- Calculating the predictions
```predict = model.predict(norm_test)```

- Calculating the accuracy
```accuracy_score(test_labels, predict)```

- Calculating the f1-score
```f1_score(test_labels, predict)```

- Print report
```print(classification_report(test_labels, predict))```

- Printing the first 10 positive and the first 10 negative words

```
weights = np.squeeze(model.coef_)

idxes = np.argsort(weights)

words = np.array(bow.words)

print('the first 10 negative words are', words[idxes[:10]])

print('the first 10 positive words are', words[idxes[-10:]])
```

