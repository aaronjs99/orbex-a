"""Finite policy rollouts and append-only run logs, independent of toy physics.

An environment supplies reset(seed=...) -> observation and step(normalized
planning preference) -> (observation,reward,terminated,truncated,info). A physical
adapter must use the maintained simulator/admission; this loop never moves a
spacecraft itself. See THIRD_PARTY_NOTICES.md for source attribution.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,os
import numpy as np
from orbexa.planning.ppo_proposal import RolloutTransition


class TrainingRun:
    """New run directories only; resume by forking from a checkpoint.

    Older CSV histories/checkpoints are never truncated to resume an experiment.
    Checkpoints contain optimizer, schedule, generator and episode state.
    """
    def __init__(self,directory,*,metadata=None,parent_checkpoint=None):
        parent=None
        if parent_checkpoint is not None:
            path=Path(parent_checkpoint).resolve()
            manifest=path.with_suffix(path.suffix+'.json')
            record=json.loads(manifest.read_text(encoding='utf-8'))
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if record.get('sha256')!=digest or record.get('bytes')!=path.stat().st_size:
                raise ValueError('Parent checkpoint identity differs from its saved manifest')
            parent=dict(path=str(path),sha256=digest,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.parent_checkpoint=parent
        record=dict(schema='orbexa.proposal-training-run.v1',utc=datetime.now(timezone.utc).isoformat(),
            metadata={} if metadata is None else metadata,parent_checkpoint=self.parent_checkpoint)
        (self.directory/'run.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        self.step_path=self.directory/'steps.csv';self.episode_path=self.directory/'episodes.csv'
        self.step_columns=('episode','step','reward','intervention_norm','terminated','truncated')
        self.episode_columns=('episode','reward','steps','terminated','truncated','learning_rate','entropy','value_loss')
        self._append(self.step_path,self.step_columns,None);self._append(self.episode_path,self.episode_columns,None)

    @staticmethod
    def _append(path,columns,row):
        with path.open('a',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns)
            if row is None:writer.writeheader()
            else:writer.writerow(row)
            stream.flush();os.fsync(stream.fileno())

    def step(self,row):self._append(self.step_path,self.step_columns,row)
    def episode(self,row):self._append(self.episode_path,self.episode_columns,row)

    def plot(self,output_path):
        """Plot actual recorded rewards, interventions and learning metrics."""
        import matplotlib.pyplot as plt
        with self.episode_path.open(newline='',encoding='utf-8') as stream:episodes=list(csv.DictReader(stream))
        with self.step_path.open(newline='',encoding='utf-8') as stream:steps=list(csv.DictReader(stream))
        if not episodes:raise ValueError('No completed training episodes to plot')
        figure,axes=plt.subplots(3,1,figsize=(8,8),constrained_layout=True)
        e=[int(r['episode']) for r in episodes]
        axes[0].plot(e,[float(r['reward']) for r in episodes]);axes[0].set_ylabel('Episode return')
        axes[1].plot(e,[float(r['entropy']) for r in episodes]);axes[1].set_ylabel('Policy entropy estimate')
        axes[2].plot(range(len(steps)),[float(r['intervention_norm']) for r in steps])
        axes[2].set_ylabel('Proposal intervention norm');axes[2].set_xlabel('Recorded step')
        for axis in axes[:2]:axis.set_xlabel('Completed episode')
        for axis in axes:axis.grid(alpha=.2)
        figure.suptitle('Planning-policy training — recorded environment scope applies')
        figure.savefig(output_path,dpi=180);plt.close(figure)
        return Path(output_path)


def train_policy(policy,environment,run,*,episodes,max_steps):
    """One bounded on-policy update per completed/truncated episode."""
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (episodes,max_steps)):
        raise ValueError('Declare finite positive episode and step limits')
    (run.directory/'policy-config.json').write_text(json.dumps(asdict(policy.config),indent=2)+'\n',encoding='utf-8')
    summaries=[]
    for _ in range(episodes):
        episode=policy.episodes_completed;observation=environment.reset(seed=policy.config.seed+episode)
        records=[];total=0.
        for step in range(max_steps):
            sample=policy.sample(observation)
            next_observation,reward,terminated,truncated,info=environment.step(sample.action.copy())
            if type(terminated) is not bool or type(truncated) is not bool or not np.isfinite(reward):
                raise ValueError('Environment must return finite rewards and explicit termination/truncation events')
            truncated=truncated or (step==max_steps-1 and not terminated)
            next_value=0. if terminated else policy.value(next_observation)
            records.append(RolloutTransition(sample,float(reward),next_value,terminated,truncated))
            intervention=float(info.get('intervention_norm',0.))
            if not np.isfinite(intervention) or intervention<0:
                raise ValueError('Intervention diagnostic must be finite and nonnegative')
            run.step(dict(episode=episode,step=step,reward=float(reward),intervention_norm=intervention,
                terminated=terminated,truncated=truncated))
            total+=reward;observation=next_observation
            if terminated or truncated:break
        metrics=policy.update(records);policy.episodes_completed+=1
        summary=dict(episode=episode,reward=float(total),steps=step+1,terminated=terminated,truncated=truncated,
            learning_rate=metrics['learning_rate'],entropy=metrics['entropy'],value_loss=metrics['value_loss'])
        run.episode(summary);summaries.append(summary)
        policy.save_checkpoint(run.directory/f'checkpoint-{policy.episodes_completed:08d}.pt')
    return summaries
