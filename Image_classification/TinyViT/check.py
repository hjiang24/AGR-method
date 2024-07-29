import pickle

with open("dic_record.pkl","rb") as file:
    data = pickle.load(file)
print(max(data["accuracy_val"]))