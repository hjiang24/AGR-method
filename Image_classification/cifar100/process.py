import pickle
import pandas as pd

# Step 1: Load data from pickle file
with open('result_18_AdamW.pkl', 'rb') as f:
    data = pickle.load(f)
    AdamW_loss = data['train']
    test = []
    for acc in data["test"]:
        test.append(round(acc.item()*100,2))
    AdamW_acc = test
with open('result_18_AdamW_AGR.pkl', 'rb') as f:
    data = pickle.load(f)
    AdamW_AGR_loss = data['train']
    test = []
    for acc in data["test"]:
        test.append(round(acc.item()*100,2))
    AdamW_AGR_acc = test
# Step 2: Prepare data (example assuming 'data' is a list of dictionaries)

data = {'AdamW':AdamW_loss,
        'AdamW_AGR':AdamW_AGR_loss}
df = pd.DataFrame(data,index=[i for i in range(200)])
excel_file_path = "Resnet18_loss.xlsx"
df.to_excel(excel_file_path , engine='openpyxl',index=False)

data = {'AdamW':AdamW_acc,
        'AdamW_AGR':AdamW_AGR_acc}
df = pd.DataFrame(data,index=[i for i in range(200)])
excel_file_path = "Resnet18_acc.xlsx"
df.to_excel(excel_file_path , engine='openpyxl',index=False)
