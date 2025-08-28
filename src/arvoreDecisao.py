import pandas as pd
from sklearn import datasets, tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pydotplus
from IPython.display import Image

# importa a base de dados iris
iris = datasets.load_iris()

X, y = iris.data, iris.target
class_names = iris.target_names

# Particiona a base de dados
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.15)

# Árvore de decisão
tree_iris = DecisionTreeClassifier(
    random_state=0,
    criterion='entropy',
    class_weight={0: 1, 1: 1, 2: 1}  # ajustado para os 3 grupos do iris
)
tree_iris = tree_iris.fit(X_train, y_train)

print("Acurácia (base de treinamento):", tree_iris.score(X_train, y_train))

y_pred = tree_iris.predict(X_test)
print("Acurácia de previsão:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cnf_matrix = confusion_matrix(y_test, y_pred)
cnf_table = pd.DataFrame(
    data=cnf_matrix,
    index=iris.target_names,
    columns=[x + " (prev)" for x in iris.target_names]
)
print(cnf_table)

from dtreeviz import DecisionTreeViz

viz = DecisionTreeViz(
    tree_iris,
    X_train,
    y_train,
    target_name="espécie",
    feature_names=iris.feature_names,
    class_names=["setosa", "versicolor", "virginica"]
)
viz.save("tree_iris.svg")

# Visualização com pydotplus + graphviz
dot_data = tree.export_graphviz(
    tree_iris,
    out_file=None,
    rounded=True,
    filled=True,
    feature_names=iris.feature_names,
    class_names=["setosa", "versicolor", "virginica"]
)

graph = pydotplus.graph_from_dot_data(dot_data)

# Mostrar imagem (funciona em Jupyter/Colab, não em terminal puro)
Image(graph.create_png())
