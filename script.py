import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
#1

breast_cancer_data = load_breast_cancer()

#2
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)

#3
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

#4+5+6 Splitting the data into Training and Validation Sets
training_data, validation_data, training_labels, validation_labels =train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state = 100)

#7 Splitting the data into Training and Validation Sets
print(len(training_data))
print(len(training_labels))

#8

#classifier = KNeighborsClassifier(n_neighbors = 3)

#9+10+11 Train your classifier using the fit function, Score = 0.9473684210526315 -> Good

#classifier.fit(training_data, training_labels)
#print(classifier.score(validation_data,validation_labels))

#9+10+11 Train your classifier using the fit function, Score = 0.9473684210526315 -> Good

#classifier = KNeighborsClassifier(n_neighbors = 3)

#9+10+11 Train your classifier using the fit function, Score = 0.9473684210526315 -> Good

k_list =range(1,101)
accuracies =[]

for k in range(1,101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data,validation_labels))

#16
plt.plot(k_list,accuracies)
plt.xlabel("K")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()
