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
        self.fc_out = nn.Linear(hidden_sizes[1], 70)  

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


def factorized_mse(preds, targets, partitions):
    losses = []
    for i in range(len(partitions) - 1):
        start, end = partitions[i], partitions[i + 1]
        mse = F.mse_loss(preds[:, start:end], targets[:, start:end], reduction="mean")
        losses.append(mse)
    return torch.mean(torch.stack(losses))

def train_model_mse(model, X_train, y_train, X_val, y_val, partitions, batch_size=512, epochs=100, lr=1e-3, device=None):

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


def train_decoder(algo, skill_dim, save_path, obs_list=None, phi_list= None):
    partitions = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70]

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

    model = Decoder(skill_dim=skill_dim)
    model = train_model_mse(model, X_train, y_train, X_val, y_val, partitions, batch_size=1024, epochs=100, lr=1e-3)

    save_model(model, save_path)


def rollouts(algo):
    skill_dim = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if algo == "susd":
        option_policy_checkpoint_path = f'final_models/particle/SUSD/option_policy10000.pt'
        traj_encoder_checkpoint_path = f'final_models/particle/SUSD/traj_encoder10000.pt'
        skill_dim = 20 # N=10 & d=2

    elif algo == "metra": 
        option_policy_checkpoint_path = 'final_models/particle/METRA/option_policy10000.pt'    
        traj_encoder_checkpoint_path = 'final_models/particle/METRA/traj_encoder10000.pt'

    elif algo == "csd":
        option_policy_checkpoint_path = 'final_models/particle/CSD/option_policy10000.pt'    
        traj_encoder_checkpoint_path = 'final_models/particle/CSD/traj_encoder10000.pt'

    elif algo == "lsd":
        option_policy_checkpoint_path = 'final_models/particle/LSD/option_policy10000.pt'    
        traj_encoder_checkpoint_path = 'final_models/particle/LSD/traj_encoder10000.pt'

    elif algo == "diayn":
        option_policy_checkpoint_path = 'final_models/particle/DIAYN/option_policy10000.pt'    
        traj_encoder_checkpoint_path = 'final_models/particle/DIAYN/traj_encoder10000.pt'

    # Load checkpoints
    option_ckpt = torch.load(option_policy_checkpoint_path)
    traj_ckpt = torch.load(traj_encoder_checkpoint_path)
    option_policy = option_ckpt["policy"].to(device).eval()
    traj_encoder = traj_ckpt["traj_encoder"].to(device).eval()

    obs_list, phi_list = [], []
    done, steps = True, 0
    z_period = 200
    env = create_particle_env()

    # tqdm progress bar
    with tqdm(total=100000, desc=f"Rollouts ({algo})") as pbar:
        while steps < 100000:
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

# algo = "susd"
# train_decoder(algo, 20, save_path=f"results/decoder/{algo}.pth")


# algo = "metra"
# train_decoder(algo, 2, save_path=f"results/decoder/{algo}.pth")


algo = "lsd"
skill_dim, obs_list, phi_list = rollouts(algo=algo)
train_decoder(algo, skill_dim, save_path=f"results/decoder/{algo}.pth", obs_list=obs_list, phi_list=phi_list)