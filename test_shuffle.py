from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
seed_everything(42)
data = [1,2,3,4,5,6,7,8,9,10]
dataloader = DataLoader(data, batch_size=2, shuffle=True)
for time in range(10):
    for batch in dataloader:
        print(batch)
        break
