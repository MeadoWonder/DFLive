from sklearn import metrics


with open('./data/results.txt', 'r') as f:
    results = f.readlines()

labels = []
preds = []
for r in results:
    idx = r.find(' ')
    labels.append(int(r[idx+1:idx+2]))
    preds.append(float(r[0:idx]))

roc = metrics.roc_auc_score(labels, preds)
print(roc)
