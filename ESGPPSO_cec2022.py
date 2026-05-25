import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opfunu.cec_based.cec2022 import *


class CVAE(nn.Module):
    def __init__(self, target_dim, latent_dim, condition_dim):
        super(CVAE, self).__init__()
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

    def encode(self, x, c):
        inputs = torch.cat([c, x], dim=1)
        h = self.encoder(inputs)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([c, z], dim=1)
        return self.decoder(inputs)

    def forward(self, condition, target):
        mu, logvar = self.encode(target, condition)
        z = self.reparameterize(mu, logvar)
        recon_target = self.decode(z, condition)
        return recon_target, mu, logvar

class PSO():
    def __init__(self, fun_num):
        self.Search_Agents = 50
        self.dimensions = 10            # 10D 20D
        self.Max_iteration = 200       
        self.max_fes = 10000            
        self.Uppernound = 100
        self.Lowerbound = -100
        self.v_min = -2
        self.v_max = 2
        self.fun_num = fun_num
        self.CEC2022 = [F12022(self.dimensions), F22022(self.dimensions), F32022(self.dimensions),
                        F42022(self.dimensions), F52022(self.dimensions), F62022(self.dimensions),
                        F72022(self.dimensions),
                        F82022(self.dimensions), F92022(self.dimensions), F102022(self.dimensions),
                        F112022(self.dimensions), F122022(self.dimensions)]
        self.H = 10
        self.stagnation_threshold = 3
        self.latent_dim = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = f"./tra_data_cec2022/pso/et_models/{self.dimensions}D/et_cvae_pso_f{self.fun_num}.pth"

    def _normalize_pop(self, pop, p_min, p_max):
        return (pop - p_min) / (p_max - p_min + 1e-8)

    def _denormalize_pop(self, pop_norm, p_min, p_max):
        return pop_norm * (p_max - p_min + 1e-8) + p_min

    def _normalize_fit(self, fit, f_min, f_max):
        log_fit = np.log1p(fit)
        return (log_fit - f_min) / (f_max - f_min + 1e-8)

    def fitness_count(self, x):
        fit = self.CEC2022[self.fun_num - 1].evaluate(x)

        return fit

    def check_velocity(self, v):
        return np.clip(v, self.v_min, self.v_max)

    def check_position(self, x):
        return np.clip(x, self.Lowerbound, self.Uppernound)

    def predict_cvae(self, pop, fitness):
        if not os.path.exists(self.model_path):
            return None

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            norm_pop = self._normalize_pop(pop, checkpoint['pop_min'], checkpoint['pop_max'])
            norm_fit = self._normalize_fit(fitness, checkpoint['log_fit_min'], checkpoint['log_fit_max'])

            condition_np = np.concatenate([norm_pop.flatten(), norm_fit.flatten()]).reshape(1, -1)
            condition_tensor = torch.FloatTensor(condition_np).to(self.device)

            model = CVAE(
                condition_dim=condition_tensor.shape[1],
                target_dim=condition_tensor.shape[1],
                latent_dim=checkpoint['latent_dim']
            ).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            with torch.no_grad():
                z = torch.randn(1, checkpoint['latent_dim']).to(self.device)
                recon_target = model.decode(z, condition_tensor)

            recon_pop_norm = recon_target[0, :self.Search_Agents * self.dimensions].cpu().numpy()
            cvae_pop_1d = self._denormalize_pop(recon_pop_norm, checkpoint['pop_min'], checkpoint['pop_max'])
            cvae_pop = cvae_pop_1d.reshape(self.Search_Agents, self.dimensions)
            return self.check_position(cvae_pop)

        except Exception as e:
            print(f"CVAE Error: {e}")
            return None

    def PSO_searcher(self, run_index):
        np.random.seed(2026 + self.fun_num + run_index)
        PSOer = np.random.uniform(self.Lowerbound, self.Uppernound, (self.Search_Agents, self.dimensions))
        V = np.zeros((self.Search_Agents, self.dimensions))
        np.random.seed(None)

        PSOerFitness = np.array([self.fitness_count(ind) for ind in PSOer])
        Pbest = PSOer.copy()
        PbestFitness = PSOerFitness.copy()

        best_idx = np.argmin(PbestFitness)
        Score = PbestFitness[best_idx]
        BestPSOer = Pbest[best_idx].copy()

        Convergence = np.full(self.max_fes, Score)
        FEcount = self.Search_Agents
        Convergence[:FEcount] = Score

        Memory_c1, Memory_c2 = np.ones(self.H) * 2.0, np.ones(self.H) * 2.0
        m_idx_c1, m_idx_c2 = 0, 0
        stagnation_counter = 0

        for t in range(self.Max_iteration):
            if FEcount >= self.max_fes: break
            w =  0.5
            improved_global = False
            S_c1, S_c2 = [], []

            for r in range(self.Search_Agents):
                if FEcount >= self.max_fes: break

                idx_c1 = np.random.randint(self.H)
                idx_c2 = np.random.randint(self.H)

                c1_r = np.random.normal(Memory_c1[idx_c1], 0.1)
                c2_r = np.random.normal(Memory_c2[idx_c2], 0.1)

                c1_r = np.clip(c1_r, 0.0, 4.0)
                c2_r = np.clip(c2_r, 0.0, 4.0)

                r1, r2 = np.random.rand(self.dimensions), np.random.rand(self.dimensions)
                V[r] = w * V[r] + c1_r * r1 * (Pbest[r] - PSOer[r]) + c2_r * r2 * (BestPSOer - PSOer[r])
                V[r] = self.check_velocity(V[r])
                PSOer[r] = self.check_position(PSOer[r] + V[r])
                fitness = self.fitness_count(PSOer[r])
                FEcount += 1

                if fitness < PbestFitness[r]:
                    Pbest[r] = PSOer[r].copy()
                    PbestFitness[r] = fitness
                    S_c1.append(c1_r)
                    S_c2.append(c2_r)

                if fitness < Score:
                    Score = fitness
                    BestPSOer = PSOer[r].copy()
                    improved_global = True

                if FEcount <= self.max_fes: Convergence[FEcount - 1] = Score
            if len(S_c1) > 0:
                Memory_c1[m_idx_c1] = np.mean(S_c1)
                m_idx_c1 = (m_idx_c1 + 1) % self.H
            if len(S_c2) > 0:
                Memory_c2[m_idx_c2] = np.mean(S_c2)
                m_idx_c2 = (m_idx_c2 + 1) % self.H

            if not improved_global:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter >= self.stagnation_threshold:
                cvae_pop = self.predict_cvae(PSOer, PSOerFitness)
                if cvae_pop is not None:
                    cvae_fitness = np.array([self.fitness_count(ind) for ind in cvae_pop])

                    combined_pop = np.vstack([PSOer, cvae_pop])
                    combined_fit = np.concatenate([PSOerFitness, cvae_fitness])
                    best_indices = np.argsort(combined_fit)[:self.Search_Agents]
                    new_PSOer = np.zeros((self.Search_Agents, self.dimensions))
                    new_V = np.zeros((self.Search_Agents, self.dimensions))
                    new_fit = np.zeros(self.Search_Agents)
                    new_Pbest = np.zeros((self.Search_Agents, self.dimensions))
                    new_PbestFitness = np.zeros(self.Search_Agents)

                    for i, idx in enumerate(best_indices):
                        new_PSOer[i] = combined_pop[idx]
                        new_fit[i] = combined_fit[idx]

                        if idx < self.Search_Agents:
                            new_V[i] = V[idx].copy()
                            new_Pbest[i] = Pbest[idx].copy()
                            new_PbestFitness[i] = PbestFitness[idx]
                        else:
                            v_ind = np.random.randint(0, self.Search_Agents)
                            new_V[i] = V[v_ind]* np.random.uniform(1.0, 2.0)
                            new_Pbest[i] = combined_pop[idx].copy()
                            new_PbestFitness[i] = combined_fit[idx]
                    PSOer, V, PSOerFitness, Pbest, PbestFitness = new_PSOer, new_V, new_fit, new_Pbest, new_PbestFitness
                    curr_best_idx = np.argmin(PSOerFitness)
                    if PSOerFitness[curr_best_idx] < Score:
                        Score = PSOerFitness[curr_best_idx]
                        BestPSOer = PSOer[curr_best_idx].copy()
                        if FEcount <= self.max_fes:
                            Convergence[FEcount - 1] = Score
                    stagnation_counter = 0

            if FEcount >= self.max_fes: break
        return Score, BestPSOer, Convergence

    def PSO_ReRun(self, Recount):
        score_sum = np.zeros(self.max_fes)
        total_score = np.zeros((Recount, self.max_fes))

        for i in range(Recount):
            b_score, b_psoer, score_count = self.PSO_searcher(i)
            score_sum += score_count
            total_score[i] = score_count
        avg_scorecount = score_sum / Recount

        plt.figure()
        x = np.arange(0, self.max_fes, 1)
        plt.semilogy(x, avg_scorecount)
        plt.title(f'Convergence Curve of Function {self.fun_num}')
        plt.xlabel("# of fitness evaluations")
        plt.ylabel("Log Average Fitness Score")
        plt.grid(True, which="both", ls="--")

        dir_path_pic = f'./MC2022Result/{self.dimensions}D/PicResult/'
        os.makedirs(dir_path_pic, exist_ok=True)
        plt.savefig(dir_path_pic + f'score_f{self.fun_num}.png')
        plt.close()

        return avg_scorecount, total_score