import os
import json

fine_tuned_models = f"./fine_tuned_models_with_extra"
subfolders = [f.name for f in os.scandir(fine_tuned_models) if f.is_dir()]

checkpoints = []
for folder in subfolders:
    print(folder)
    batchsize = int(folder[56:58])
    print(batchsize)
    subfolders1 = [f.name for f in os.scandir(f"{fine_tuned_models}/{folder}") if f.is_dir()]
    tmpList = []
    for line2 in subfolders1:
        if line2.startswith('checkpoint'):
            tmpList.append(int(line2.split('-')[1]))
    tmpList.sort(reverse=True)
    print("saved checkpoints:")
    print([f"checkpoint-{str(i)}" for i in tmpList])
    
    if batchsize == 16:
        cpDir10 = f"./{folder}/checkpoint-{str(tmpList[0])}"
        cpDir5 = f"./{folder}/checkpoint-{str(tmpList[5])}"
        cpDir0 = f"./{folder}/checkpoint-{str(tmpList[9])}"
    elif batchsize == 32:
        cpDir10 = f"./{folder}/checkpoint-{str(tmpList[0])}"
        cpDir5 = f"./{folder}/checkpoint-{str(tmpList[5])}"
        cpDir0 = f"./{folder}/checkpoint-{str(tmpList[9])}"
    
    checkpoints.append([cpDir0, cpDir5, cpDir10])
    print("selected checkpoints:")
    print([cpDir0, cpDir5, cpDir10])
    print()
json.dump(checkpoints, open(f"./selected_checkpoints.json", "w", encoding="utf-8"))