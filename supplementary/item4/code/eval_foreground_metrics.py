"""Compute full-image and foreground-mask RGB PSNR/SSIM/LPIPS-Alex/DISTS.

The foreground mask is derived only from the HR reference: the largest
bright connected component is selected, morphologically closed, and dilated
slightly. The same mask is used for every method and is saved for audit.
"""
import argparse, csv, json
from pathlib import Path
import cv2, numpy as np
from PIL import Image
import torch
import lpips
from DISTS_pytorch import DISTS


EXTS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}
def imgs(root): return {p.stem:p for p in Path(root).iterdir() if p.is_file() and p.suffix.lower() in EXTS}
def read(p): return np.asarray(Image.open(p).convert('RGB'),dtype=np.uint8)
def psnr(a,b):
    x=(a.astype(np.float64)-b.astype(np.float64)); m=float(np.mean(x*x)); return float('inf') if m==0 else float(10*np.log10(255.0**2/m))
def ssim_map(a,b):
    a=a.astype(np.float64); b=b.astype(np.float64); k=cv2.getGaussianKernel(11,1.5); w=k@k.T
    m1=cv2.GaussianBlur(a,(11,11),1.5); m2=cv2.GaussianBlur(b,(11,11),1.5)
    v1=cv2.GaussianBlur(a*a,(11,11),1.5)-m1*m1; v2=cv2.GaussianBlur(b*b,(11,11),1.5)-m2*m2; co=cv2.GaussianBlur(a*b,(11,11),1.5)-m1*m2
    c1=(.01*255)**2; c2=(.03*255)**2
    return ((2*m1*m2+c1)*(2*co+c2)/((m1*m1+m2*m2+c1)*(v1+v2+c2))).astype(np.float64)
def mask_from_hr(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    # Seed is bright and chromatic against the nearly black substrate.
    th=max(28.0,float(np.percentile(gray,70))*0.28)
    m=((gray>=th)&((hsv[...,1]>=18)|(gray>=65))).astype(np.uint8)
    ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,ker)
    n,lab,stats,_=cv2.connectedComponentsWithStats(m,8)
    if n<=1: return (m>0)
    idx=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])); out=(lab==idx).astype(np.uint8)
    out=cv2.morphologyEx(out,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    out=cv2.dilate(out,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=1)
    return out.astype(bool)
def tensor(x,device): return torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).float().div_(127.5).sub_(1).unsqueeze(0).to(device)
def save_mask(path,m): Image.fromarray((m.astype(np.uint8)*255)).save(path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--hr_dir',required=True); ap.add_argument('--lr_dir',required=True); ap.add_argument('--out_dir',required=True); ap.add_argument('--crop',type=int,default=8); ap.add_argument('--batch',type=int,default=8); ap.add_argument('--method',action='append',required=True,help='NAME=DIR; use DIR=__BICUBIC__ for LR bicubic'); args=ap.parse_args()
    hr=imgs(args.hr_dir); lr=imgs(args.lr_dir); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); maskdir=out/'masks'; maskdir.mkdir(exist_ok=True)
    masks={}; cover=[]
    for n,p in hr.items():
        m=mask_from_hr(read(p)); masks[n]=m; cover.append(float(m.mean())); save_mask(maskdir/(n+'.png'),m)
    methods={}
    for spec in args.method:
        name,path=spec.split('=',1); methods[name]=path
    bicdir=out/'bicubic_generated'; bicdir.mkdir(exist_ok=True)
    if any(v=='__BICUBIC__' for v in methods.values()):
        for n,p in lr.items():
            if n in hr: Image.fromarray(read(p)).resize((read(hr[n]).shape[1],read(hr[n]).shape[0]),Image.Resampling.BICUBIC).save(bicdir/(n+'.png'))
        methods={k:(str(bicdir) if v=='__BICUBIC__' else v) for k,v in methods.items()}
    dev='cuda' if torch.cuda.is_available() else 'cpu'; print('device',dev,flush=True)
    lper=lpips.LPIPS(net='alex',verbose=False).eval().to(dev); dists=DISTS().eval().to(dev)
    allsummary=[]
    for name,path in methods.items():
        pred=imgs(path); names=sorted(set(hr)&set(pred)); rows=[]; accum=[]
        for st in range(0,len(names),args.batch):
            ss=[]; gg=[]; batchrows=[]
            for n in names[st:st+args.batch]:
                g=read(hr[n]); s=read(pred[n]);
                if s.shape!=g.shape: raise RuntimeError(f'{name} size mismatch {n}: {s.shape} vs {g.shape}')
                c=args.crop; gm=g[c:-c,c:-c] if c else g; sm=s[c:-c,c:-c] if c else s; mm=masks[n][c:-c,c:-c] if c else masks[n]; mm=mm.astype(bool)
                if not np.any(mm): raise RuntimeError(f'empty foreground mask {n}')
                roi_ps=psnr(sm[mm],gm[mm]); roi_ss=float(np.mean([ssim_map(sm[...,i],gm[...,i])[mm] for i in range(3)]))
                # Outside the common HR foreground, make the two images identical.
                smask=sm.copy(); smask[~mm]=gm[~mm]
                ss.append(tensor(smask,dev)); gg.append(tensor(gm,dev))
                full_ss=float(np.mean([ssim_map(sm[...,i],gm[...,i])[5:-5,5:-5].mean() for i in range(3)]))
                batchrows.append({'name':n,'mask_fraction':float(mm.mean()),'full_psnr_rgb':psnr(sm,gm),'full_ssim_rgb':full_ss,'fg_psnr_rgb':roi_ps,'fg_ssim_rgb':roi_ss})
            with torch.inference_mode():
                la=lper(torch.cat(ss),torch.cat(gg),normalize=True).flatten().detach().cpu().numpy(); ds=dists(torch.cat(ss),torch.cat(gg)).flatten().detach().cpu().numpy()
            for r,a,d in zip(batchrows,la,ds): r['fg_lpips_alex']=float(a); r['fg_dists']=float(d); rows.append(r)
            print(name, min(st+len(batchrows),len(names)),'/',len(names),flush=True)
        md=out/name; md.mkdir(exist_ok=True)
        fields=['name','mask_fraction','full_psnr_rgb','full_ssim_rgb','fg_psnr_rgb','fg_ssim_rgb','fg_lpips_alex','fg_dists']
        with (md/'per_image.csv').open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
        sm={'method':name,'count':len(rows),'mean_mask_fraction':float(np.mean([r['mask_fraction'] for r in rows]))}
        for k in fields[2:]:
            v=np.asarray([r[k] for r in rows],float); sm[k]=float(v.mean()); sm[k+'_std']=float(v.std(ddof=1)) if len(v)>1 else 0.0
        with (md/'summary.json').open('w',encoding='utf-8') as f: json.dump(sm,f,ensure_ascii=False,indent=2)
        allsummary.append(sm)
    fields=['method','count','mean_mask_fraction','full_psnr_rgb','full_psnr_rgb_std','full_ssim_rgb','full_ssim_rgb_std','fg_psnr_rgb','fg_psnr_rgb_std','fg_ssim_rgb','fg_ssim_rgb_std','fg_lpips_alex','fg_lpips_alex_std','fg_dists','fg_dists_std']
    with (out/'full_vs_foreground_summary.csv').open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(allsummary)
    with (out/'README.txt').open('w',encoding='utf-8') as f:
        f.write('Foreground evaluation\n'); f.write('Foreground masks are generated from each HR image only, then reused for every method. Full metrics use crop=8. Foreground PSNR/SSIM are computed only on mask pixels; LPIPS-Alex and DISTS set pixels outside the common mask to the HR reference so background does not contribute.\n')
    print('saved',out,flush=True)
if __name__=='__main__': main()
