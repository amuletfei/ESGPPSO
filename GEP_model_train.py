import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class PSODataset(Dataset):
    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
        self.pop_min, self.pop_max = self._get_range("int_pop")
        self.fitness_min, self.fitness_max = self._get_range("int_fit")
        self.log_fit_min, self.log_fit_max = self._get_log_fitness_range()

        self.cached_data = []
        for file in tqdm(self.file_list, desc="Loading Data"):
            data = np.load(os.path.join(self.data_dir, file))
            initial_pop = self._normalize(data["int_pop"], self.pop_min, self.pop_max)
            final_pop = self._normalize(data["final_pop"], self.pop_min, self.pop_max)
            log_int_fit = np.log1p(data["int_fit"])
            initial_fitness = self._normalize(log_int_fit, self.log_fit_min, self.log_fit_max)
            log_final_fit = np.log1p(data["final_fit"])
            final_fitness = self._normalize(log_final_fit, self.log_fit_min, self.log_fit_max)
            condition = np.concatenate([initial_pop.flatten(), initial_fitness.flatten()], axis=0)
            target = np.concatenate([final_pop.flatten(), final_fitness.flatten()], axis=0)
            self.cached_data.append((torch.FloatTensor(condition), torch.FloatTensor(target)))

    def _get_range(self, key):
        min_val = np.inf
        max_val = -np.inf
        for file in self.file_list:
            data = np.load(os.path.join(self.data_dir, file))
            arr = data[key]
            min_val = min(min_val, arr.min())
            max_val = max(max_val, arr.max())
        return min_val, max_val

    def _get_log_fitness_range(self):
        min_val = np.inf
        max_val = -np.inf
        for file in self.file_list:
            data = np.load(os.path.join(self.data_dir, file))
            arr_int = np.log1p(data["int_fit"])
            arr_final = np.log1p(data["final_fit"])
            min_val = min(min_val, arr_int.min(), arr_final.min())
            max_val = max(max_val, arr_int.max(), arr_final.max())
        return min_val, max_val

    def _normalize(self, arr, min_val, max_val):
        return (arr - min_val) / (max_val - min_val + 1e-8)  

    def _denormalize(self, arr, min_val, max_val):
        return arr * (max_val - min_val) + min_val

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        return self.cached_data[idx]


class CVAE(nn.Module):
    def __init__(self, condition_dim=550, target_dim=550, latent_dim=64):
        super(CVAE, self).__init__()
        self.condition_dim = condition_dim  
        self.target_dim = target_dim  
        self.latent_dim = latent_dim  
        self.encoder = nn.Sequential(
            nn.Linear(condition_dim + target_dim, 1024),  
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)  
        self.fc_logvar = nn.Linear(256, latent_dim)  
        self.decoder = nn.Sequential(
            nn.Linear(condition_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, target_dim),
            nn.Sigmoid()  
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(std)  
        return mu + eps * std  

    def forward(self, condition, target):
        x = torch.cat([condition, target], dim=1)  
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        z_cond = torch.cat([condition, z], dim=1)  
        recon_target = self.decoder(z_cond)

        return recon_target, mu, logvar

def loss_function(recon_target, target, condition, mu, logvar, current_epoch, agents, dim):
    recon_pop = recon_target[:, :-agents]
    recon_fit = recon_target[:, -agents:]
    real_pop = target[:, :-agents]
    real_fit = target[:, -agents:]
    int_fit = condition[:, -agents:]
    improvement = int_fit - real_fit
    weight_mask = torch.where(improvement > 0,
                              torch.tensor(2.0, device=target.device),
                              torch.tensor(0.5, device=target.device))
    weight_mask_pop = torch.repeat_interleave(weight_mask, repeats=dim, dim=1)
    loss_pop = torch.mean(weight_mask_pop * (recon_pop - real_pop) ** 2)
    loss_fit = torch.mean(weight_mask * (recon_fit - real_fit) ** 2)
    recon_loss = loss_pop + 200.0 * loss_fit
    kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    kl_loss = kl_loss / recon_target.size(0)
    if current_epoch < 50:
        beta = 0.0
    else:
        beta = min(0.0001, 0.0001 * (current_epoch - 50) / 100.0)

    return recon_loss + beta * kl_loss, loss_pop.item(), loss_fit.item()


def train_cvae(data_dir, input_dim, output_dim, agents, dim,epochs, batch_size=32, latent_dim=64, save_path="cvae_et_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device：{device}")
    dataset = PSODataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    model = CVAE(condition_dim=input_dim,
                 target_dim=output_dim,
                 latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_pop_loss = 0
        total_fit_loss = 0
        for batch_idx, (condition, target) in enumerate(dataloader):
            condition, target = condition.to(device), target.to(device)
            if torch.rand(1).item() < 0.2: 
                condition_noisy = torch.zeros_like(condition)
            else:
                condition_noisy = condition + torch.randn_like(condition) * 0.01           
            recon_target, mu, logvar = model(condition_noisy, target)
            loss, l_pop, l_fit = loss_function(recon_target, target, condition, mu, logvar, current_epoch=epoch,
                                              agents=agents, dim=dim)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pop_loss += l_pop
            total_fit_loss += l_fit

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{EPOCHS} | avg_loss: {total_loss / len(dataloader):.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "latent_dim": latent_dim,
        "pop_min": dataset.pop_min,
        "pop_max": dataset.pop_max,
        "log_fit_min": dataset.log_fit_min,
        "log_fit_max": dataset.log_fit_max
    }, save_path)
    print(f"et_model saved to {save_path}")
    del dataset, dataloader, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    DIM = 10  
    AGENTS = 50  
    EPOCHS = 300
    BATCH_SIZE = 32
    LATENT_DIM = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_condition_dim = AGENTS * DIM + AGENTS
    current_target_dim = AGENTS * DIM + AGENTS
    # cec2022 f1-f12
    func_list = [nu for nu in range(1, 13)]
    SAVE_DIR = f"./tra_data_cec2022/pso/et_models/{DIM}D/"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for nu in func_list:
        DATA_DIR = f'./tra_data_cec2022/pso/{DIM}D/50itr/f{nu}'
        SAVE_PATH = os.path.join(SAVE_DIR, f"et_cvae_pso_f{nu}.pth")

        if not os.path.exists(DATA_DIR):
            print(f"skip f{nu}：dir {DATA_DIR} not exists")
            continue
        print(f"training start f{nu}，{EPOCHS}  Epoch...")
        train_cvae(
            data_dir=DATA_DIR,
            input_dim=current_condition_dim,
            output_dim=current_target_dim,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            latent_dim=LATENT_DIM,
            save_path=SAVE_PATH,
            agents = AGENTS,
            dim = DIM
        )

    print("\n all finished")