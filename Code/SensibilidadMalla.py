import os,re,math,numpy as np,pandas as pd,matplotlib.pyplot as plt
try:
    from scipy.spatial import cKDTree as KDTree; HAVE_SCIPY=True
except: HAVE_SCIPY=False

ROOT="AnsysData"; SUBDIR="CarbopolSolution"; DT=0.01; OUT_BASE="mesh_sensitivity"; BRUTE_CHUNK=20000
METHODS={"HexSweep":[4,5]}

def ensure_dir(p): 
    if not os.path.isdir(p): os.makedirs(p,exist_ok=True)
def list_files(method,size):
    base=os.path.join(ROOT,method,f"Size{size}",SUBDIR)
    if not os.path.isdir(base): return []
    rx=re.compile(rf"^{re.escape(method)}{size}-(\d+)$"); out=[]
    for f in os.listdir(base):
        m=rx.match(f)
        if m:
            step=int(m.group(1)); path=os.path.join(base,f)
            try: ct=os.path.getctime(path)
            except: ct=os.path.getmtime(path)
            out.append((path,step,ct))
    return sorted(out,key=lambda t:t[1])
def read_snapshot(path):
    cols=["node","x","y","z","ux","uy","uz"]
    df=pd.read_csv(path,header=None,names=cols,skipinitialspace=True)
    df=df[pd.to_numeric(df["node"],errors="coerce").notnull()].astype(float)
    return df.reset_index(drop=True)
def nn_idx(A,B):
    if HAVE_SCIPY and len(B)>0:
        tree=KDTree(B);_,idx=tree.query(A,k=1,workers=-1);return idx
    idx=np.empty(len(A),int)
    for i in range(0,len(A),BRUTE_CHUNK):
        j=i+BRUTE_CHUNK;diff=A[i:j,None,:]-B[None,:,:]
        idx[i:j]=np.argmin(np.sum(diff*diff,axis=2),axis=1)
    return idx
def rmse_sym(Axyz,Av,Bxyz,Bv):
    i=nn_idx(Axyz,Bxyz);r1=math.sqrt(np.mean(np.sum((Av-Bv[i])**2,axis=1)))
    i=nn_idx(Bxyz,Axyz);r2=math.sqrt(np.mean(np.sum((Bv-Av[i])**2,axis=1)))
    return 0.5*(r1+r2)
def speed_stats(v):
    s=np.linalg.norm(v,axis=1);return float(np.mean(s)),float(np.std(s)),float(np.max(s))
def runtime(files):
    if len(files)<2:return 0.0,0.0,[]
    t=[c for _,_,c in files];d=np.diff(t);d=d[d>0]
    return (float(np.median(d)) if len(d) else 0.0,float(np.sum(d)) if len(d) else 0.0,d.tolist())
def title2(m,s):return f"{m}\n{s}"

def compare(method,a,b):
    out_dir=os.path.join(OUT_BASE,method,f"{a}mm_vs_{b}mm");ensure_dir(out_dir)
    fa,fb=list_files(method,a),list_files(method,b)
    if not fa or not fb:print(f"[{method}] faltan Size{a} o {b}");return
    ma={s:(p,c) for (p,s,c) in fa};mb={s:(p,c) for (p,s,c) in fb}
    steps=sorted(set(ma)&set(mb))
    if not steps:print(f"[{method}] sin pasos comunes {a} vs {b}");return
    rows=[]
    for s in steps:
        t=(s-1)*DT;dfa,dfb=read_snapshot(ma[s][0]),read_snapshot(mb[s][0])
        Axyz,Av=dfa[["x","y","z"]].to_numpy(),dfa[["ux","uy","uz"]].to_numpy()
        Bxyz,Bv=dfb[["x","y","z"]].to_numpy(),dfb[["ux","uy","uz"]].to_numpy()
        mA,sdA,MXA=speed_stats(Av);mB,sdB,MXB=speed_stats(Bv);r=rmse_sym(Axyz,Av,Bxyz,Bv)
        i=nn_idx(Axyz,Bxyz);sA=np.linalg.norm(Av,axis=1);sBn=np.linalg.norm(Bv[i],axis=1)
        linf=float(np.max(np.abs(sA-sBn)));lmean=float(np.mean(np.abs(sA-sBn)))
        rows.append({"step":s,"time_s":t,f"mean_speed_{a}mm":mA,f"std_speed_{a}mm":sdA,f"max_speed_{a}mm":MXA,
                     f"mean_speed_{b}mm":mB,f"std_speed_{b}mm":sdB,f"max_speed_{b}mm":MXB,
                     "rmse_vel_sym_mps":r,"linf_speed_diff_mps":linf,"lmean_speed_diff_mps":lmean})
    df=pd.DataFrame(rows).sort_values("step");csv_path=os.path.join(out_dir,f"{method}_{a}vs{b}.csv");df.to_csv(csv_path,index=False)
    medA,totA,dA=runtime(fa);medB,totB,dB=runtime(fb)
    plt.figure();plt.plot(df["time_s"],df["rmse_vel_sym_mps"]);plt.xlabel("t(s)");plt.ylabel("RMSE (m/s)")
    plt.title(title2(f"{method}: RMSE {a} vs {b}","Independencia de malla"));plt.grid();plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"01_rmse.png"),dpi=200)
    plt.figure();plt.plot(df["time_s"],df[f"mean_speed_{a}mm"]);plt.plot(df["time_s"],df[f"mean_speed_{b}mm"])
    plt.title(title2(f"{method}: Rapidez media","Consistencia global"));plt.legend([f"{a}mm",f"{b}mm"]);plt.grid();plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"02_mean.png"),dpi=200)
    plt.figure();plt.plot(df["time_s"],df[f"std_speed_{a}mm"]);plt.plot(df["time_s"],df[f"std_speed_{b}mm"])
    plt.title(title2(f"{method}: Dispersión rapidez","Variabilidad"));plt.legend([f"{a}mm",f"{b}mm"]);plt.grid();plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"03_std.png"),dpi=200)
    plt.figure();plt.plot(df["time_s"],df[f"max_speed_{a}mm"]);plt.plot(df["time_s"],df[f"max_speed_{b}mm"])
    plt.title(title2(f"{method}: Pico rapidez","Sensibilidad picos"));plt.legend([f"{a}mm",f"{b}mm"]);plt.grid();plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"04_max.png"),dpi=200)
    plt.figure();plt.plot(df["time_s"],df["linf_speed_diff_mps"]);plt.title(title2(f"{method}: Dif L∞","Outliers locales"))
    plt.grid();plt.tight_layout();plt.savefig(os.path.join(out_dir,"05_linf.png"),dpi=200)
    plt.figure();plt.plot(df["time_s"],df["lmean_speed_diff_mps"]);plt.title(title2(f"{method}: Dif promedio","Sesgo global"))
    plt.grid();plt.tight_layout();plt.savefig(os.path.join(out_dir,"06_lmean.png"),dpi=200)
    if len(dA)>0 or len(dB)>0:
        plt.figure(); 
        if len(dA)>0:plt.plot(np.arange(len(dA)),dA)
        if len(dB)>0:plt.plot(np.arange(len(dB)),dB)
        plt.title(title2(f"{method}: Runtime","Costo computacional"));plt.grid();plt.tight_layout()
        plt.savefig(os.path.join(out_dir,"07_runtime.png"),dpi=200)
    print(f"[{method}] {a} vs {b}: pasos {len(steps)}, RMSE medio {df['rmse_vel_sym_mps'].mean():.6g} m/s")

def main():
    ensure_dir(OUT_BASE)
    for m,sizes in METHODS.items():
        s=sorted(set(sizes))
        for i in range(len(s)-1):compare(m,s[i],s[i+1])

if __name__=="__main__":main()
