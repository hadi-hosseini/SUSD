import torch.nn as nn
import torch
from torch import distributions as pyd
import math
from torch.distributions.utils import _standard_normal


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))



class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu



def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)



class Actor(nn.Module):
	def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, sac, log_std_bounds, domain):
		super().__init__()

		self.sac = sac
		feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

		self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
								   nn.LayerNorm(feature_dim), nn.Tanh())

		policy_layers = []
		policy_layers += [
			nn.Linear(feature_dim, hidden_dim),
			nn.ReLU(inplace=True)
		]
		# add additional hidden layer for pixels
		if obs_type == 'pixels':
			policy_layers += [
				nn.Linear(hidden_dim, hidden_dim),
				nn.ReLU(inplace=True)
			]

		if self.sac:
			policy_layers += [nn.Linear(hidden_dim, 2 * action_dim)]
		else:
			policy_layers += [nn.Linear(hidden_dim, action_dim)]

		self.policy = nn.Sequential(*policy_layers)
		self.log_std_bounds = log_std_bounds

		self.domain = domain

		self.apply(weight_init)

	def forward(self, obs, std=None):
		h = self.trunk(obs)

		if self.sac:
			mu, log_std = self.policy(h).chunk(2, dim=-1)

			# constrain log_std inside [log_std_min, log_std_max]
			log_std = torch.tanh(log_std)
			log_std_min, log_std_max = self.log_std_bounds
			log_std = log_std_min + 0.5 * (log_std + 1) * (log_std_max - log_std_min)
			std = log_std.exp()
			dist = SquashedNormal(mu, std)
		else:
			mu = self.policy(h)
			mu = torch.tanh(mu)
			std = torch.ones_like(mu) * std
			dist = TruncatedNormal(mu, std)
		return dist
