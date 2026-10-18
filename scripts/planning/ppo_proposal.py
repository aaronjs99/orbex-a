"""Optional CPU PPO for bounded planning preferences, not applied spacecraft control.

PyTorch is imported only when constructing/loading a policy. Derived from the
MIT Playground master and feature/rl-cbf; see THIRD_PARTY_NOTICES.md. Actions
use a tanh-transformed Gaussian with matching likelihoods, not clipped normals.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
from pathlib import Path
import hashlib,json,math,os
import numpy as np


@dataclass(frozen=True)
class PPOConfig:
    observation_dimension: int
    action_dimension: int = 3
    hidden_size: int = 64
    learning_rate: float = 3.e-4
    final_learning_rate: float = 1.e-5
    schedule_updates: int = 500
    gamma: float = .99
    gae_lambda: float = .95
    clip: float = .2
    epochs: int = 4
    entropy_weight: float = .01
    gradient_limit: float = .5
    initial_std: float = .2
    seed: int = 0

    def __post_init__(self):
        for key in ('observation_dimension','action_dimension','hidden_size','schedule_updates','epochs'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f'{key} must be a positive integer')
        if isinstance(self.seed,bool) or not isinstance(self.seed,int) or not 0<=self.seed<2**63-1:
            raise ValueError('PPO seed must be a nonnegative signed-64-bit integer')
        for key in ('learning_rate','final_learning_rate','gradient_limit','initial_std'):
            if not np.isfinite(getattr(self,key)) or getattr(self,key)<=0:
                raise ValueError(f'{key} must be positive and finite')
        if (self.final_learning_rate>self.learning_rate or not 0<self.gamma<=1
                or not 0<=self.gae_lambda<=1 or not 0<self.clip<1
                or not np.isfinite(self.entropy_weight) or self.entropy_weight<0):
            raise ValueError('Invalid PPO discount, schedule, clip or entropy setting')
        if not math.exp(-8)<=self.initial_std<=math.exp(2):
            raise ValueError('Initial exploration std exceeds the supported log-std range')


@dataclass(frozen=True)
class PolicySample:
    observation: np.ndarray
    latent: np.ndarray
    action: np.ndarray
    log_probability: float
    value: float
    policy_fingerprint: str


@dataclass(frozen=True)
class RolloutTransition:
    sample: PolicySample
    reward: float
    next_value: float
    terminated: bool
    truncated: bool


def generalized_advantage_estimate(records,gamma,gae_lambda):
    """True termination removes bootstrap; truncation stops cross-episode GAE."""
    if not records or not 0<gamma<=1 or not 0<=gae_lambda<=1:
        raise ValueError('GAE requires a nonempty rollout and valid discounts')
    advantages=np.empty(len(records));tail=0.
    for k in reversed(range(len(records))):
        row=records[k]
        if not np.all(np.isfinite((row.reward,row.next_value,row.sample.value))):
            raise ValueError('Rollout rewards and values must be finite')
        if type(row.terminated) is not bool or type(row.truncated) is not bool:
            raise ValueError('Termination and truncation must be explicit Boolean events')
        td=row.reward+gamma*(not row.terminated)*row.next_value-row.sample.value
        tail=td+gamma*gae_lambda*(not (row.terminated or row.truncated))*tail
        advantages[k]=tail
    returns=advantages+np.array([row.sample.value for row in records])
    return advantages,returns


class PPOProposalPolicy:
    """On-policy reference learner with complete episode-boundary checkpoints.

    No training/evaluation result here constitutes a robust-control certificate.
    The seed/optimizer/schedule/RNG state is preserved; environment continuation
    is reproducible only when its reset(seed=...) contract is reproducible.
    """
    def __init__(self,config:PPOConfig):
        if not isinstance(config,PPOConfig):raise TypeError('Use a typed PPOConfig')
        try:
            import torch
        except ImportError as error:
            raise ImportError('Install ORBEX-A learning extras to construct PPO proposals') from error
        self.torch,self.config=torch,config
        class Model(torch.nn.Module):
            def __init__(model):
                super().__init__()
                def network(output):
                    return torch.nn.Sequential(torch.nn.Linear(config.observation_dimension,config.hidden_size),
                        torch.nn.Tanh(),torch.nn.Linear(config.hidden_size,config.hidden_size),torch.nn.Tanh(),
                        torch.nn.Linear(config.hidden_size,output))
                model.actor=network(config.action_dimension);model.critic=network(1)
                model.log_std=torch.nn.Parameter(torch.full((config.action_dimension,),math.log(config.initial_std)))
        # Preserve the caller's global RNG state; this policy owns a CPU generator.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed);self.model=Model().float()
        self.generator=torch.Generator(device='cpu').manual_seed(config.seed+1)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=config.learning_rate)
        self.updates=0;self.episodes_completed=0;self.exploration_resets=[]

    def _observation(self,value):
        a=np.asarray(value,dtype=np.float32)
        if a.shape!=(self.config.observation_dimension,) or not np.all(np.isfinite(a)):
            raise ValueError('PPO observation has invalid shape/values')
        return self.torch.from_numpy(a.copy())

    def _distribution(self,observations):
        mean=self.model.actor(observations)
        std=self.model.log_std.clamp(-8.,2.).exp().expand_as(mean)
        return self.torch.distributions.Normal(mean,std)

    def _log_probability(self,distribution,latent):
        # log(1-tanh(z)^2) in a stable form, including large |z|.
        jacobian=2*(math.log(2.)-latent-self.torch.nn.functional.softplus(-2*latent))
        return (distribution.log_prob(latent)-jacobian).sum(dim=-1)

    def sample(self,observation):
        torch=self.torch;x=self._observation(observation)
        with torch.no_grad():
            distribution=self._distribution(x)
            latent=distribution.loc+distribution.scale*torch.randn(distribution.loc.shape,generator=self.generator)
            action=torch.tanh(latent);probability=self._log_probability(distribution,latent)
            value=self.model.critic(x).squeeze(-1)
        return PolicySample(x.numpy().copy(),latent.numpy().copy(),action.numpy().copy(),
            float(probability),float(value),self.fingerprint())

    def predict(self,observation):
        with self.torch.no_grad():
            return self.torch.tanh(self.model.actor(self._observation(observation))).numpy().copy()

    def value(self,observation):
        with self.torch.no_grad():
            return float(self.model.critic(self._observation(observation)).squeeze(-1))

    def reset_exploration(self,std):
        if not np.isfinite(std) or not math.exp(-8)<=std<=math.exp(2):
            raise ValueError('Exploration std must lie inside the supported positive log-std range')
        with self.torch.no_grad():self.model.log_std.fill_(math.log(std))
        self.exploration_resets.append(dict(after_updates=self.updates,std=float(std)))

    def update(self,records):
        torch=self.torch;cfg=self.config
        if any(row.sample.policy_fingerprint!=self.fingerprint() for row in records):
            raise ValueError('PPO update requires rollouts from the current policy snapshot')
        advantages,returns=generalized_advantage_estimate(records,cfg.gamma,cfg.gae_lambda)
        observations=torch.stack([self._observation(r.sample.observation) for r in records])
        latent=torch.as_tensor(np.stack([r.sample.latent for r in records]),dtype=torch.float32)
        old_probability=torch.tensor([r.sample.log_probability for r in records],dtype=torch.float32)
        if latent.shape!=(len(records),cfg.action_dimension) or not torch.isfinite(latent).all() or not torch.isfinite(old_probability).all():
            raise ValueError('PPO rollout has invalid latent actions or old likelihoods')
        adv=torch.as_tensor(advantages,dtype=torch.float32)
        adv=(adv-adv.mean())/(adv.std(unbiased=False)+1.e-8)
        target=torch.as_tensor(returns,dtype=torch.float32)
        for _ in range(cfg.epochs):
            distribution=self._distribution(observations)
            log_probability=self._log_probability(distribution,latent)
            ratio=(log_probability-old_probability).exp()
            actor_loss=-torch.minimum(ratio*adv,ratio.clamp(1-cfg.clip,1+cfg.clip)*adv).mean()
            value_loss=(self.model.critic(observations).squeeze(-1)-target).square().mean()
            # Reparameterized Monte Carlo entropy of the transformed distribution.
            draw=distribution.loc+distribution.scale*torch.randn(distribution.loc.shape,generator=self.generator)
            entropy=-self._log_probability(distribution,draw).mean()
            loss=actor_loss+.5*value_loss-cfg.entropy_weight*entropy
            if not torch.isfinite(loss):raise ValueError('PPO loss is nonfinite; no successful update is reported')
            self.optimizer.zero_grad();loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(self.model.parameters(),cfg.gradient_limit,error_if_nonfinite=True)
            self.optimizer.step()
        self.updates+=1
        lr=cfg.learning_rate+(cfg.final_learning_rate-cfg.learning_rate)*min(1.,self.updates/cfg.schedule_updates)
        for group in self.optimizer.param_groups:group['lr']=lr
        return dict(actor_loss=float(actor_loss.detach()),value_loss=float(value_loss.detach()),
            entropy=float(entropy.detach()),gradient_norm=float(gradient),learning_rate=float(lr),updates=self.updates)

    def fingerprint(self):
        digest=hashlib.sha256(json.dumps(asdict(self.config),sort_keys=True).encode())
        for key,value in sorted(self.model.state_dict().items()):
            digest.update(key.encode());digest.update(value.detach().numpy().tobytes())
        return digest.hexdigest()

    def save_checkpoint(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        manifest=path.with_suffix(path.suffix+'.json')
        if path.exists() or manifest.exists():raise FileExistsError('Checkpoint destinations must be new')
        payload=dict(schema='orbexa.ppo-proposal-checkpoint.v1',config=asdict(self.config),
            model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),generator=self.generator.get_state(),
            updates=self.updates,episodes_completed=self.episodes_completed,exploration_resets=self.exploration_resets)
        with path.open('xb') as stream:
            self.torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        record=dict(schema=payload['schema'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            bytes=path.stat().st_size,updates=self.updates,episodes_completed=self.episodes_completed,
            policy_fingerprint=self.fingerprint(),scope='Planning proposal learner; no mission-safety qualification')
        with manifest.open('x',encoding='utf-8') as stream:
            json.dump(record,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        return record

    @classmethod
    def load_checkpoint(cls,path):
        path=Path(path);manifest=path.with_suffix(path.suffix+'.json')
        if manifest.stat().st_size>1048576:raise ValueError('Checkpoint manifest exceeds the finite metadata budget')
        record=json.loads(manifest.read_text(encoding='utf-8'))
        if (record.get('schema')!='orbexa.ppo-proposal-checkpoint.v1' or path.stat().st_size!=record.get('bytes')
                or hashlib.sha256(path.read_bytes()).hexdigest()!=record.get('sha256')):
            raise ValueError('Checkpoint is incomplete or its identity changed')
        import torch
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('schema')!=record['schema']:raise ValueError('Unknown checkpoint schema')
        policy=cls(PPOConfig(**payload['config']))
        policy.model.load_state_dict(payload['model']);policy.optimizer.load_state_dict(payload['optimizer'])
        policy.generator.set_state(payload['generator']);policy.updates=payload['updates']
        policy.episodes_completed=payload['episodes_completed'];policy.exploration_resets=payload['exploration_resets']
        if policy.fingerprint()!=record['policy_fingerprint']:
            raise ValueError('Restored policy identity differs from the committed checkpoint')
        return policy


@dataclass(frozen=True)
class PPOInitialGuess:
    """Map a deterministic bounded policy to nominal MPC seeds in declared units.

    Encoder: [state(6), reference-state(6)] / observation_scale. Training must use
    exactly that same frame/clock/normalization contract. This does not assert a
    learned policy has been trained on the nonlinear mission.
    """
    policy: PPOProposalPolicy
    observation_scale: tuple[float,...] = (1.,)*12
    encoder: object | None = None

    def propose(self,problem):
        from orbexa.planning.trajectory_hints import AffinePrediction,TrajectoryHint,input_limits,hint_identity
        if problem.num_states!=6 or problem.num_inputs!=3 or self.policy.config.action_dimension!=3:
            raise ValueError('PPO MPC adapter requires six states and three commands')
        if self.encoder is None and self.policy.config.observation_dimension!=12:
            raise ValueError('The default PPO feature contract has twelve entries')
        if self.encoder is not None and not callable(getattr(self.encoder,'encode',None)):
            raise ValueError('A custom feature encoder must implement encode(state,reference,k)')
        scale=np.asarray(self.observation_scale,dtype=float)
        if scale.shape!=(12,) or not np.all(np.isfinite(scale)) or np.any(scale<=0):
            raise ValueError('Observation scales must be a finite positive twelve-vector')
        lo,hi=input_limits(problem)
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise ValueError('Normalized learned preferences need explicit finite command bounds')
        model=AffinePrediction.from_problem(problem);x=model.initial_state.copy();policy_id=self.policy.fingerprint()
        controls=np.empty((3,problem.num_steps-1))
        for k in range(controls.shape[1]):
            observation=(np.r_[x,problem.reference_states[:,k]-x]/scale if self.encoder is None else
                         self.encoder.encode(x.copy(),problem.reference_states[:,k].copy(),k))
            action=self.policy.predict(observation)
            controls[:,k]=(hi+lo)/2+(hi-lo)*action/2
            x=model.state_matrices[k] @ x+model.input_matrices[k] @ controls[:,k]+model.offsets[k]
        if self.policy.fingerprint()!=policy_id:
            raise ValueError('Policy changed during the native initial-guess construction')
        states=model.rollout(controls);method='deterministic-ppo-mpc-initial-guess-v1'
        return TrajectoryHint(states,controls,dict(method=method,policy_fingerprint=policy_id,
            observation_scale=scale.tolist(),identity=hint_identity(states,controls,method),
            role='Initial guess only; no learned policy can bypass native planning or physical admission'))
