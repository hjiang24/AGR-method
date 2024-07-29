import pickle
import torch

with open("record_adamw.pkl","rb") as file:
    data = pickle.load(file)
    print(min(data["train"]))
    print(max(data["test"]))


with open("record_adam.pkl","rb") as file:
    data = pickle.load(file)
    print(min(data["train"]))
    print(max(data["test"]))

with open("record_coefficient_adamw.pkl","rb") as file:
    data = pickle.load(file)
    print(min(data["train"]))
    print(max(data["test"]))
# a = torch.tensor([5,4])
# b = a.detach()
# b = b-1
# print(a)