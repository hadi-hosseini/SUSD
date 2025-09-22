import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


from pettingzoo.mpe import simple_heterogenous_v3
from pettingzoo.utils.wrappers.centralized_wrapper import CentralizedWrapper
from envs.mp.particle import Particle



# MP: OBS_SPACE = 70, SKILL_DIM FOR BASELINES ARE 2 AND FOR SUSD IS 20
class Decoder(nn.Module):
    def __init__(self, skill_dim, hidden_sizes=(35, 70)):
        super().__init__()
        self.fc1 = nn.Linear(skill_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def factorized_mse(preds, targets, partitions):
    losses = []
    for i in range(len(partitions) - 1):
        start, end = partitions[i], partitions[i + 1]
        mse = F.mse_loss(preds[:, start:end], targets[:, start:end], reduction="mean")
        losses.append(mse)
    return torch.mean(torch.stack(losses))

def train_model_mse(model, X_train, y_train, X_val, y_val, partitions, batch_size=512, epochs=100, lr=1e-4, device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = factorized_mse(preds, yb, partitions)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_dataset)

        # Print results
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}, Train MSE: {epoch_loss:.6f}, Val MSE: {val_loss:.6f}")

    return model


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


def load_model(d: int, path: str, hidden_sizes=(35, 70), device=None):
    model = Decoder(d=d, hidden_sizes=hidden_sizes, is_mse=is_mse).to(device)

    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model weights loaded from {path}")
    return model


def train_decoder(algo, save_path, hidden_sizes, partitions, skill_dim=2, obs_list=None, phi_list= None):

    if obs_list is None:
        phi_list = np.load(f"results/decoder/data/phi_list_{algo}.npy")
        obs_list = np.load(f"results/decoder/data/obs_list_{algo}.npy")

    X_train, X_val, y_train, y_val = train_test_split(phi_list, obs_list, test_size=0.2, random_state=42, shuffle=True)
    X_train = torch.from_numpy(X_train).float()
    X_val   = torch.from_numpy(X_val).float()
    y_train = torch.from_numpy(y_train).float()
    y_val   = torch.from_numpy(y_val).float()

    print("Train shape:", X_train.shape, y_train.shape)
    print("Validation shape:", X_val.shape, y_val.shape)

    model = Decoder(skill_dim=skill_dim, hidden_sizes=hidden_sizes)
    model = train_model_mse(model, X_train, y_train, X_val, y_val, partitions, batch_size=1024, epochs=100, lr=1e-3)

    save_model(model, save_path)


def rollouts(algo, env_name, skill_dim=2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if algo == "susd":
        option_policy_checkpoint_path = f'final_models/{env_name}/SUSD/option_policy10000.pt'
        traj_encoder_checkpoint_path = f'final_models/{env_name}/SUSD/traj_encoder10000.pt'

    elif algo == "metra": 
        option_policy_checkpoint_path = f'final_models/{env_name}/METRA/option_policy10000.pt'    
        traj_encoder_checkpoint_path = f'final_models/{env_name}/METRA/traj_encoder10000.pt'

    elif algo == "csd":
        option_policy_checkpoint_path = f'final_models/{env_name}/CSD/option_policy10000.pt'    
        traj_encoder_checkpoint_path = f'final_models/{env_name}/CSD/traj_encoder10000.pt'

    elif algo == "lsd":
        option_policy_checkpoint_path = f'final_models/{env_name}/LSD/option_policy10000.pt'    
        traj_encoder_checkpoint_path = f'final_models/{env_name}/LSD/traj_encoder10000.pt'

    elif algo == "diayn":
        option_policy_checkpoint_path = f'final_models/{env_name}/DIAYN/option_policy10000.pt'    
        traj_encoder_checkpoint_path = f'final_models/{env_name}/DIAYN/traj_encoder10000.pt'

    # Load checkpoints
    option_ckpt = torch.load(option_policy_checkpoint_path)
    traj_ckpt = torch.load(traj_encoder_checkpoint_path)
    option_policy = option_ckpt["policy"].to(device).eval()
    traj_encoder = traj_ckpt["traj_encoder"].to(device).eval()

    obs_list, phi_list = [], []
    done, steps = True, 0
    z_period = 200

    if env_name == "particle":
        env = create_particle_env()
    elif env_name == "elden_kitchen":
        env = create_elden_env()
    elif env_name == "gunner":
        env = create_gunner_env()
    elif env_name == "ant":
        env = create_ant_env()
    elif env_name == "half_cheetah":
        env = create_half_cheetah()

    # tqdm progress bar
    with tqdm(total=1e5, desc=f"Rollouts ({algo})") as pbar: # 10000000
        while steps < 1e5:
            if done:
                obs = env.reset()
                done = False
                random_z = np.random.randn(1, skill_dim)
                random_z /= np.linalg.norm(random_z)
                random_z = torch.tensor(random_z, dtype=torch.float32).to(device)
            else:
                if steps % z_period == 0:
                    random_z = np.random.randn(1, skill_dim)
                    random_z /= np.linalg.norm(random_z)
                    random_z = torch.tensor(random_z, dtype=torch.float32).to(device)
                    obs = env.reset()  # RESET EACH 200 STEPS

                obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                input_tensor = torch.cat([obs, random_z], dim=-1)

                with torch.no_grad():
                    if algo == "susd":
                        phi = traj_encoder(obs)
                    else:
                        phi = traj_encoder(obs).mean
                    action_np, _ = option_policy.get_action(input_tensor)
                action = action_np[0]

                obs_list.append(obs.squeeze(0).cpu().numpy())
                phi_list.append(phi.squeeze(0).cpu().numpy())

                obs, _, done, info = env.step(action)
                steps += 1
                pbar.update(1)  # update tqdm bar

    phi_list = np.array(phi_list)
    obs_list = np.array(obs_list)

    np.save(f"results/decoder/data/phi_list_{algo}.npy", phi_list)
    np.save(f"results/decoder/data/obs_list_{algo}.npy", obs_list)

    return skill_dim, obs_list, phi_list


def create_particle_env():
    distances = list(range(0, 10))       # 0–9
    agent_info = list(range(10, 50))     # 10–49
    station_info = list(range(50, 70))   # 50–69

    custom_order = []

    for i in range(10):
        custom_order.append(distances[i])                       
        custom_order.extend(agent_info[i*4:(i+1)*4])           
        custom_order.extend(station_info[i*2:(i+1)*2])

    env = simple_heterogenous_v3.parallel_env(
            render_mode= "rgb_array",
            max_cycles=1000,
            continuous_actions=True,
            local_ratio=0,
            N=10,
            img_encoder=None)

    env = CentralizedWrapper(env, simplify_action_space=True)
    env = Particle(env, custom_order, (512, 480))
    return env


def create_elden_env(seed=0):
    from envs.elden_kitchen.elden_kitchen import elden_kitchen, EldenKitchen
    env = elden_kitchen(reward_scale=0.0, horizon=50, render=False) # reward_scale = 0.0 is used for USD
    custom_order = [113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 0, 1, 2, 3] # 29 arm + 4 don't know
    custom_order += [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 101, 102, 103, 104, 105, 106]  # 22 pot
    custom_order += [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37] # 18 butter
    custom_order += [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56] # 19 meatball
    custom_order += [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 107, 108, 109, 110, 111, 112] # 22 button
    custom_order += [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86] # 14 stove
    custom_order += [87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100] # 14 target   
    env = EldenKitchen(env, custom_order=custom_order) 
    return env

def create_gunner_env(seed=0):
    from envs.moma_2d.moma_2d_gym_env import MoMa2DGymEnv
    custom_order = [0, 1, 2, 3, 12, 13,
                        4, 5, 6, 7, 14, 15, 16,
                        8, 9, 10, 11, 17] # base, arm, view (ORIGINAL)
    env = MoMa2DGymEnv(max_step=1000, custom_order=custom_order)
    env.reset()
    return env

def create_ant_env(seed=0):
    from envs.mujoco.ant_env import AntEnv
    env = AntEnv(render_hw=100)
    return env

def create_half_cheetah(seed=0):
    from envs.mujoco.half_cheetah_env import HalfCheetahEnv
    env = HalfCheetahEnv(render_hw=100)
    return env


algo = "lsd"
skill_dim = 2
env_name = "elden_kitchen"

if env_name == "particle":
    hidden_sizes = (35, 70)
    partitions = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
    if algo == "susd":
        skill_dim = 20

elif env_name == "elden_kitchen":
    partitions = [0, 33, 55, 73, 92, 114, 128, 142]
    hidden_sizes = (64, 142)
    if algo == "susd":
        skill_dim = 14

elif env_name == "gunner":
    partitions = [0, 6, 13, 18]
    hidden_sizes = (9, 18)
    if algo == "susd":
        skill_dim = 6

elif env_name == "half_cheetah":
    partitions = [0, 18]
    hidden_sizes = (9, 18)
    if algo == "susd":
        skill_dim = 2

elif env_name == "ant":
    partitions = [0, 29]
    hidden_sizes = (10, 29)
    if algo == "susd":
        skill_dim = 2


# skill_dim, obs_list, phi_list = rollouts(algo=algo, env_name=env_name, skill_dim=skill_dim)
train_decoder(algo=algo, skill_dim=skill_dim, hidden_sizes=hidden_sizes, partitions=partitions, save_path=f"results/decoder/{algo}.pth")