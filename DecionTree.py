import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import pydotplus
from IPython.display import Image

# Prepare the dataset
data = {
    "Rain": ["No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No"],
    "Sprinkler": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "No"],
    "Grass": ["Dry", "Wet", "Dry", "Wet", "Wet", "Wet", "Dry", "Wet", "Dry", "Dry", "Dry", "Dry", "Dry", "Wet", "Wet", "Dry"]
}

df = pd.DataFrame(data)

# Perform one-hot encoding on the categorical columns
df = pd.get_dummies(df, columns=["Rain", "Sprinkler"])

# Train the decision tree
X = df.drop(columns=["Grass"])
y = df["Grass"]

clf = DecisionTreeClassifier()
clf.fit(X, y)

# Visualize the decision tree
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=y.unique(),
                                filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

# Save the decision tree image to a file
graph.write_png("decision_tree.png")

# Display the image
Image("decision_tree.png")