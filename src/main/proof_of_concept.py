from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split

linnerud = load_linnerud()

X_train, X_test, y_train, y_test = train_test_split(linnerud['data'], linnerud['target'], random_state=0)