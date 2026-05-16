import csv

def load_iris(path="IRIS.csv"):
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        for row in reader:
            sepal_len = float(row[0])
            sepal_wid = float(row[1])
            petal_len = float(row[2])
            petal_wid = float(row[3])
            species = row[4]
            data.append([sepal_len, sepal_wid, petal_len, petal_wid, species])
    return data


if __name__ == "__main__":
    iris = load_iris()
    print(f"Loaded {len(iris)} samples")
    print("First 3 rows:", iris[:3])
