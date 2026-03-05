from data.datatsets import get_testload
from models import get_cnn
import torch 
import pandas as pd 
testload = get_testload()


GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]


subs = []
model = get_cnn()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(
    "/kaggle/input/datasets/alokmanawat/cnnmodel-1/cnn-1.pth",
    map_location=torch.device("cpu")
))

model.eval()
with torch.no_grad():
    for song , fid in testload:
        song = song.to(device)
        out = model(song)
        preds = torch.argmax(out, dim=1)
        preds = preds.cpu().numpy()
        for i in range(len(fid)):
            subs.append((fid[i],preds[i]))

cleaned = [(int(fid.item()), GENRES[int(pred)]) for fid, pred in subs]

df = pd.DataFrame(cleaned, columns=["id", "label"])