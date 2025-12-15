import time,math,threading,queue
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import ray
import gfootball.env as football_env

FEATURE_DIM=93
OBS_DIM=115

class FeatureEngineer:
    GOAL_POS=np.array([1.0,0.0],dtype=np.float32)
    OWN_GOAL_POS=np.array([-1.0,0.0],dtype=np.float32)
    GOAL_TOP=np.array([1.0,0.044],dtype=np.float32)
    GOAL_BOTTOM=np.array([1.0,-0.044],dtype=np.float32)
    @staticmethod
    def extract(obs):
        sq=False
        if obs.ndim==1:obs=obs.reshape(1,-1);sq=True
        B=obs.shape[0]
        o=obs[:,:115] if obs.shape[1]>=115 else np.pad(obs,((0,0),(0,115-obs.shape[1])))
        f=np.zeros((B,FEATURE_DIM),dtype=np.float32)
        lp=o[:,0:22].reshape(B,11,2);ld=o[:,22:44].reshape(B,11,2)
        rp=o[:,44:66].reshape(B,11,2);rd=o[:,66:88].reshape(B,11,2)
        bp=o[:,88:90];bz=o[:,90:91];bd=o[:,91:94]
        bo=o[:,94:97];bot=np.argmax(bo,axis=1)-1
        ai=np.clip(o[:,97].astype(np.int32),0,10)
        gm=o[:,98:105];st=o[:,105:115]
        bi=np.arange(B);ap=lp[bi,ai];bs=np.linalg.norm(bd[:,:2],axis=1)
        f[:,0]=bp[:,0];f[:,1]=bp[:,1];f[:,2]=np.clip(bz[:,0],0,1)
        f[:,3]=np.clip(bs,0,2);f[:,4]=bd[:,0];f[:,5]=bd[:,1];f[:,6]=bot/2.0
        rb=bp-ap;db=np.linalg.norm(rb,axis=1);ab=np.arctan2(rb[:,1],rb[:,0])
        f[:,7]=rb[:,0];f[:,8]=rb[:,1];f[:,9]=np.clip(db,0,2);f[:,10]=ab/np.pi
        gv=FeatureEngineer.GOAL_POS-bp;dg=np.linalg.norm(gv,axis=1)
        ga=np.abs(np.arctan2(gv[:,1],gv[:,0]))
        vt=FeatureEngineer.GOAL_TOP-bp;vb=FeatureEngineer.GOAL_BOTTOM-bp
        sa=np.abs(np.arctan2(vt[:,1],vt[:,0])-np.arctan2(vb[:,1],vb[:,0]))
        dog=np.linalg.norm(FeatureEngineer.OWN_GOAL_POS-bp,axis=1)
        f[:,11]=np.clip(dg,0,2);f[:,12]=ga/np.pi;f[:,13]=sa/np.pi
        f[:,14]=(dg<0.35).astype(np.float32);f[:,15]=np.clip(dog,0,2)
        rx=rp[:,:,0];ki=np.argmax(rx,axis=1);kp=rp[bi,ki]
        kd=np.linalg.norm(bp-kp,axis=1);kv=bp-kp;ka=np.arctan2(kv[:,1],kv[:,0])
        kval=kp[:,0]>0.7
        f[:,16]=np.where(kval,np.clip(kd,0,2),1.0);f[:,17]=np.where(kval,ka/np.pi,0.0)
        la=np.abs(lp[:,:,0])>0.01;ra=np.abs(rp[:,:,0])>0.01
        tr=lp-ap[:,None,:];td=np.linalg.norm(tr,axis=2)
        td[bi,ai]=999.0;td=np.where(la,td,999.0);tsi=np.argsort(td,axis=1)
        for i in range(5):
            idx=tsi[:,i];rel=lp[bi,idx]-ap;dirs=ld[bi,idx];val=td[bi,idx]<100
            f[:,18+i*4]=np.where(val,rel[:,0],0);f[:,19+i*4]=np.where(val,rel[:,1],0)
            f[:,20+i*4]=np.where(val,dirs[:,0],0);f[:,21+i*4]=np.where(val,dirs[:,1],0)
        opr=rp-ap[:,None,:];opd=np.linalg.norm(opr,axis=2)
        opd=np.where(ra,opd,999.0);osi=np.argsort(opd,axis=1)
        for i in range(5):
            idx=osi[:,i];rel=rp[bi,idx]-ap;dirs=rd[bi,idx];val=opd[bi,idx]<100
            f[:,38+i*4]=np.where(val,rel[:,0],0);f[:,39+i*4]=np.where(val,rel[:,1],0)
            f[:,40+i*4]=np.where(val,dirs[:,0],0);f[:,41+i*4]=np.where(val,dirs[:,1],0)
        bx=bp[:,0];lx=lp[:,:,0]
        tah=np.sum((lx>bx[:,None])&la,axis=1)/11.0
        dah=np.sum((rx>bx[:,None])&ra,axis=1)/11.0
        na=tah*11-dah*11
        pwd=np.linalg.norm(lp[:,:,None,:]-rp[:,None,:,:],axis=3)
        mod=np.min(np.where(ra[:,None,:],pwd,10.0),axis=2)
        fm=(mod>0.15)&la;ft=np.sum(fm,axis=1)/11.0
        lyv=np.where(la,lp[:,:,1],np.nan);ts=np.nanstd(lyv,axis=1);ts=np.nan_to_num(ts,nan=0.0)
        f[:,58]=tah;f[:,59]=dah;f[:,60]=np.clip(na/5.0,-1,1);f[:,61]=ft;f[:,62]=np.clip(ts,0,0.5)
        oda=np.linalg.norm(rp-ap[:,None,:],axis=2);oda=np.where(ra,oda,10.0)
        f[:,63]=np.clip(np.min(oda,axis=1),0,1)
        oam=(rx>bx[:,None])&ra;odah=np.where(oam,oda,10.0);sah=np.min(odah,axis=1)
        f[:,64]=np.clip(np.where(np.any(oam,axis=1),sah,1.0),0,1)
        f[:,65]=np.clip(np.sum(oda<0.2,axis=1)/3.0,0,1)
        tir=(td<0.3)&(td>0.05);f[:,66]=np.clip(np.sum(tir&fm,axis=1)/5.0,0,1)
        srx=np.sort(rx,axis=1);ofl=np.maximum(bx,srx[:,1]);ax=lp[bi,ai,0]
        f[:,67]=ofl;f[:,68]=((ax>ofl)&(bot==0)).astype(np.float32)
        f[:,69]=(bx>0.33).astype(np.float32);f[:,70]=((bx>=-0.33)&(bx<=0.33)).astype(np.float32)
        f[:,71]=(bx<-0.33).astype(np.float32);by=bp[:,1]
        f[:,72]=(by>0.2).astype(np.float32);f[:,73]=(by<-0.2).astype(np.float32)
        sds=st[:,:8];sdi=np.argmax(sds,axis=1);sda=np.any(sds>0,axis=1);sang=sdi*(2*np.pi/8)
        f[:,74]=st[:,8];f[:,75]=st[:,9]
        f[:,76]=np.where(sda,np.cos(sang),0);f[:,77]=np.where(sda,np.sin(sang),0)
        f[:,78:85]=gm;f[:,85]=0.5;f[:,86]=0.0;f[:,87]=0.0;f[:,88]=0.0;f[:,89]=1.0
        f[:,90]=sa*np.where(kval,np.clip(kd,0,1),1.0)
        f[:,91]=np.clip(bd[:,0]*bs,-1,1)
        f[:,92]=np.clip(f[:,64],0,1)*0.4+np.clip(f[:,91],0,1)*0.3+np.clip(na/5.0+0.5,0,1)*0.3
        return f[0] if sq else f

def orthogonal_init(m,gain=1.0):
    if isinstance(m,nn.Linear):
        nn.init.orthogonal_(m.weight,gain=gain)
        if m.bias is not None:nn.init.zeros_(m.bias)

class PolicyValueNet(nn.Module):
    def __init__(self,obs_dim=OBS_DIM,feat_dim=FEATURE_DIM,hidden=256,n_actions=19):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(obs_dim+feat_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.policy=nn.Sequential(nn.Linear(hidden,hidden//2),nn.ReLU(),nn.Linear(hidden//2,n_actions))
        self.value=nn.Sequential(nn.Linear(hidden,hidden//2),nn.ReLU(),nn.Linear(hidden//2,1))
        self.apply(lambda m:orthogonal_init(m,gain=np.sqrt(2)))
        orthogonal_init(self.policy[-1],gain=0.01)
        orthogonal_init(self.value[-1],gain=1.0)
    def forward(self,obs,feat):
        h=self.encoder(torch.cat([obs,feat],dim=-1))
        return self.policy(h),self.value(h).squeeze(-1)
    def get_action(self,obs,feat,deterministic=False):
        logits,value=self.forward(obs,feat)
        dist=Categorical(logits=logits)
        action=logits.argmax(dim=-1) if deterministic else dist.sample()
        return action,dist.log_prob(action),value
    def evaluate(self,obs,feat,actions):
        logits,value=self.forward(obs,feat)
        dist=Categorical(logits=logits)
        return dist.log_prob(actions),dist.entropy(),value

def compute_vtrace(behavior_log_probs,target_log_probs,actions,rewards,dones,values,bootstrap,gamma=0.99,rho_bar=1.0,c_bar=1.0):
    with torch.no_grad():
        log_rhos=(target_log_probs-behavior_log_probs).clamp(-20,20)
        rhos=torch.exp(log_rhos)
        clipped_rhos=torch.clamp(rhos,max=rho_bar)
        cs=torch.clamp(rhos,max=c_bar)
        T=len(rewards)
        vs=torch.zeros(T+1,device=values.device)
        vs[T]=bootstrap
        for t in reversed(range(T)):
            delta=clipped_rhos[t]*(rewards[t]+gamma*(1-dones[t])*values[t+1 if t+1<T else T-1]-values[t])
            vs[t]=values[t]+delta+gamma*(1-dones[t])*cs[t]*(vs[t+1]-values[t+1 if t+1<T else T-1])
        vs=vs[:-1]
        advantages=clipped_rhos*(rewards+gamma*(1-dones)*torch.cat([vs[1:],bootstrap.unsqueeze(0)])-values)
    return vs,advantages,clipped_rhos

@ray.remote
class AsyncWorker:
    def __init__(self,worker_id,num_agents=1):
        self.worker_id=worker_id
        self.num_agents=num_agents
        self.fe=FeatureEngineer()
        self.env=football_env.create_environment(env_name="11_vs_11_easy_stochastic",representation="simple115v2",
            number_of_left_players_agent_controls=num_agents,rewards="scoring,checkpoints",
            write_goal_dumps=False,write_full_episode_dumps=False,render=False,write_video=False)
        self.device=torch.device('cpu')
        self.model=PolicyValueNet().to(self.device)
        self.model.eval()
        self.obs=None
        self.feat=None
        self.ep_return=0.0
        self.ep_steps=0
        self._reset()
    def set_weights(self,weights):
        self.model.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in weights.items()})
    def _reset(self):
        raw=self.env.reset()
        self._update_obs(raw)
        self.ep_return=0.0
        self.ep_steps=0
    def _update_obs(self,raw):
        if isinstance(raw,list):raw=np.array(raw)
        if raw.ndim==1:raw=raw.reshape(1,-1)
        self.obs=raw[0][:OBS_DIM].astype(np.float32)
        self.feat=self.fe.extract(self.obs)
    def collect(self,traj_len=128):
        obs_l,feat_l,act_l,rew_l,done_l,logp_l,val_l=[],[],[],[],[],[],[]
        ep_rets,ep_wins=[],[]
        for _ in range(traj_len):
            obs_t=torch.from_numpy(self.obs).float().unsqueeze(0)
            feat_t=torch.from_numpy(self.feat).float().unsqueeze(0)
            with torch.no_grad():
                action,logp,value=self.model.get_action(obs_t,feat_t)
            act=action.item()
            env_act=act if self.num_agents==1 else [act]
            raw,reward,done,info=self.env.step(env_act)
            step_r=float(reward) if np.isscalar(reward) else float(reward[0]) if len(reward)>0 else 0.0
            self.ep_return+=step_r
            self.ep_steps+=1
            ep_done=bool(done) or self.ep_steps>=3000
            obs_l.append(self.obs.copy())
            feat_l.append(self.feat.copy())
            act_l.append(act)
            rew_l.append(step_r)
            done_l.append(float(ep_done))
            logp_l.append(logp.item())
            val_l.append(value.item())
            if ep_done:
                won=info["score"][0]>info["score"][1] if isinstance(info,dict) and "score" in info else self.ep_return>0
                ep_rets.append(self.ep_return)
                ep_wins.append(1.0 if won else 0.0)
                self._reset()
            else:
                self._update_obs(raw)
        obs_t=torch.from_numpy(self.obs).float().unsqueeze(0)
        feat_t=torch.from_numpy(self.feat).float().unsqueeze(0)
        with torch.no_grad():
            _,_,bootstrap=self.model.get_action(obs_t,feat_t)
        return {
            'obs':np.array(obs_l,dtype=np.float32),'feat':np.array(feat_l,dtype=np.float32),
            'actions':np.array(act_l,dtype=np.int64),'rewards':np.array(rew_l,dtype=np.float32),
            'dones':np.array(done_l,dtype=np.float32),'log_probs':np.array(logp_l,dtype=np.float32),
            'values':np.array(val_l,dtype=np.float32),'bootstrap':bootstrap.item(),
            'ep_returns':ep_rets,'ep_wins':ep_wins,'worker_id':self.worker_id
        }
    def close(self):
        self.env.close()

class IMPALATrainer:
    def __init__(self,num_workers=24,num_agents=1,traj_len=256,lr=3e-4,gamma=0.99,
                 rho_bar=1.0,c_bar=1.0,ent_coef=0.01,vf_coef=0.5,max_grad=0.5,
                 batch_size=4096,device="cuda",ckpt_dir="./checkpoints_impala"):
        self.num_workers=num_workers
        self.num_agents=num_agents
        self.traj_len=traj_len
        self.lr=lr
        self.gamma=gamma
        self.rho_bar=rho_bar
        self.c_bar=c_bar
        self.ent_coef=ent_coef
        self.vf_coef=vf_coef
        self.max_grad=max_grad
        self.batch_size=batch_size
        self.ckpt_dir=Path(ckpt_dir)
        self.ckpt_dir.mkdir(exist_ok=True)
        self.device=torch.device(device if torch.cuda.is_available() else "cpu")
        self.model=PolicyValueNet().to(self.device)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=lr,eps=1e-5)
        ray.init(ignore_reinit_error=True)
        self.workers=[AsyncWorker.remote(i,num_agents) for i in range(num_workers)]
        self.total_steps=0
        self.update_count=0
        self.ep_returns=deque(maxlen=100)
        self.ep_wins=deque(maxlen=100)
        self.traj_queue=queue.Queue(maxsize=num_workers*2)
        self.weight_lock=threading.Lock()
        self.running=True
        print(f"IMPALA Trainer: {self.device}, {num_workers} workers, {sum(p.numel() for p in self.model.parameters()):,} params")
    def _get_weights(self):
        return {k:v.cpu().numpy() for k,v in self.model.state_dict().items()}
    def _collector_thread(self):
        weights=self._get_weights()
        ray.get([w.set_weights.remote(weights) for w in self.workers])
        pending={w.collect.remote(self.traj_len):w for w in self.workers}
        sync_counter=0
        while self.running:
            done,_=ray.wait(list(pending.keys()),num_returns=1,timeout=0.01)
            for ref in done:
                worker=pending.pop(ref)
                try:
                    traj=ray.get(ref)
                    self.traj_queue.put(traj,timeout=1.0)
                except:pass
                pending[worker.collect.remote(self.traj_len)]=worker
                sync_counter+=1
                if sync_counter%5==0:
                    with self.weight_lock:
                        weights=self._get_weights()
                    worker.set_weights.remote(weights)
    def _process_trajectory(self,traj):
        obs=torch.from_numpy(traj['obs']).float().to(self.device)
        feat=torch.from_numpy(traj['feat']).float().to(self.device)
        actions=torch.from_numpy(traj['actions']).long().to(self.device)
        rewards=torch.from_numpy(traj['rewards']).float().to(self.device)
        dones=torch.from_numpy(traj['dones']).float().to(self.device)
        behavior_log_probs=torch.from_numpy(traj['log_probs']).float().to(self.device)
        old_values=torch.from_numpy(traj['values']).float().to(self.device)
        bootstrap=torch.tensor(traj['bootstrap'],device=self.device)
        target_log_probs,entropy,values=self.model.evaluate(obs,feat,actions)
        vtrace_targets,advantages,rhos=compute_vtrace(
            behavior_log_probs,target_log_probs,actions,rewards,dones,old_values,
            bootstrap,self.gamma,self.rho_bar,self.c_bar)
        return obs,feat,actions,vtrace_targets,advantages,entropy,values,rhos
    def train(self,total_steps=50_000_000,log_interval=10,ckpt_interval=100):
        print(f"Starting IMPALA training for {total_steps:,} steps...")
        collector=threading.Thread(target=self._collector_thread,daemon=True)
        collector.start()
        start_time=time.time()
        traj_buffer=[]
        buffer_steps=0
        while self.total_steps<total_steps:
            try:
                traj=self.traj_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            traj_buffer.append(traj)
            buffer_steps+=len(traj['obs'])
            self.total_steps+=len(traj['obs'])
            for r in traj['ep_returns']:self.ep_returns.append(r)
            for w in traj['ep_wins']:self.ep_wins.append(w)
            if buffer_steps>=self.batch_size:
                all_obs,all_feat,all_act,all_vtarget,all_adv,all_ent,all_val,all_rho=[],[],[],[],[],[],[],[]
                for t in traj_buffer:
                    o,f,a,vt,ad,en,vl,rh=self._process_trajectory(t)
                    all_obs.append(o);all_feat.append(f);all_act.append(a)
                    all_vtarget.append(vt);all_adv.append(ad);all_ent.append(en)
                    all_val.append(vl);all_rho.append(rh)
                obs=torch.cat(all_obs);feat=torch.cat(all_feat);actions=torch.cat(all_act)
                vtrace_targets=torch.cat(all_vtarget);advantages=torch.cat(all_adv)
                entropy=torch.cat(all_ent);values=torch.cat(all_val);rhos=torch.cat(all_rho)
                advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
                target_log_probs,_,new_values=self.model.evaluate(obs,feat,actions)
                policy_loss=-(target_log_probs*advantages.detach()).mean()
                value_loss=F.mse_loss(new_values,vtrace_targets.detach())
                entropy_loss=-entropy.mean()
                loss=policy_loss+self.vf_coef*value_loss+self.ent_coef*entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),self.max_grad)
                with self.weight_lock:
                    self.optimizer.step()
                self.update_count+=1
                traj_buffer=[]
                buffer_steps=0
                if self.update_count%log_interval==0:
                    elapsed=time.time()-start_time
                    sps=self.total_steps/elapsed if elapsed>0 else 0
                    wr=np.mean(self.ep_wins)*100 if self.ep_wins else 0
                    ret=np.mean(self.ep_returns) if self.ep_returns else 0
                    print(f"[{self.update_count:4d}] {self.total_steps/1e6:.2f}M | {sps/1e3:.1f}k sps | "
                          f"Win:{wr:.1f}% | Ret:{ret:.2f} | Loss:{loss.item():.3f} | "
                          f"Ent:{entropy.mean().item():.3f} | Rho:{rhos.mean().item():.2f}")
                if self.update_count%ckpt_interval==0:
                    self._save_checkpoint()
        self.running=False
        self._save_checkpoint(final=True)
        print(f"Training complete! Final win rate: {np.mean(self.ep_wins)*100:.1f}%")
    def _save_checkpoint(self,final=False):
        name="final" if final else f"update_{self.update_count}"
        path=self.ckpt_dir/f"checkpoint_{name}.pt"
        torch.save({'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
                    'steps':self.total_steps,'updates':self.update_count,
                    'win_rate':np.mean(self.ep_wins) if self.ep_wins else 0},path)
        print(f"  Saved: {path}")
    def close(self):
        self.running=False
        for w in self.workers:
            try:ray.get(w.close.remote())
            except:pass
        ray.shutdown()

def main():
    trainer=IMPALATrainer(
        num_workers=24,num_agents=1,traj_len=256,lr=3e-4,gamma=0.99,
        rho_bar=1.0,c_bar=1.0,ent_coef=0.01,vf_coef=0.5,max_grad=0.5,
        batch_size=4096,device="cuda",ckpt_dir="./checkpoints_impala")
    try:
        trainer.train(total_steps=50_000_000,log_interval=10,ckpt_interval=100)
    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        trainer.close()

if __name__=="__main__":
    main()