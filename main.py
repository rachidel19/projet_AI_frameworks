import torch
from torch.utils.data import Dataset, DataLoader
from model import NCF
import pandas as pd
import argparse


class Ratings_Datset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index()

    def __len__(self):
        return len(self.df)
  
    def __getitem__(self, idx):
        user = user2id[self.df['user_id'][idx]]
        user = torch.tensor(user, dtype=torch.long) 
        item = item2id[self.df['recipe_id'][idx]]
        item = torch.tensor(item, dtype=torch.long)
        rating = torch.tensor(self.df['rating'][idx], dtype=torch.float)
        return user, item, rating

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()

  parser.add_argument('--model_weights', type=str, default=None, metavar='N',
                    help='load weights of the model (default: None)')
  parser.add_argument('--data_test', type=str, default=None, metavar='N',
                    help='load csv file for the test  (default: None)')

  args = parser.parse_args()
  model_weights = args.model_weights
  data_test = args.data_test
  
  import torch
  cuda_enable = True
  if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU")
  else:
    device = torch.device("cpu")
    print("CPU")
    
  df = pd.read_csv(data_test)
  n_user = 12925
  n_items = 160901
  #model = NCF(n_user, n_items)
  model = torch.load(model_weights, map_location='cpu')
  model = model.to(device)
  #model.load_state_dict(torch.load(model_weights, map_location='cpu'))
  model.eval()
  
  user_list = df.user_id.unique()
  item_list = df.recipe_id.unique()
  user2id = {w: i for i, w in enumerate(user_list)}
  item2id = {w: i for i, w in enumerate(item_list)}
  test = DataLoader(Ratings_Datset(df), batch_size=10, shuffle=True ,num_workers=1)
  users, movies, r = next(iter(test))
  #users = users.cuda()
  #movies = movies.cuda()
  #r = r.cuda()

  y = model(users, movies)*5
  print("ratings", r[:10].data)
  print("predictions:", y.flatten()[:10].data)
